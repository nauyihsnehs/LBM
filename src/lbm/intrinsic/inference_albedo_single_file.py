import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize


class BaseModel(torch.nn.Module):
    def load(self, path):
        parameters = torch.load(path, map_location=torch.device("cpu"))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)


def _calc_same_pad(i, k, s, d):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4
    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained, in_chan=3, group_width=8):
    resnet = torch.hub.load(
        "facebookresearch/WSL-Images",
        f"resnext101_32x{group_width}d_wsl",
    )
    if in_chan != 3:
        resnet.conv1 = torch.nn.Conv2d(in_chan, 64, 7, 2, 3, bias=False)
    return _make_resnet_backbone(resnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2])
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])
    return pretrained


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False, in_chan=3):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
    )
    if in_chan != 3:
        efficientnet.conv_stem = Conv2dSame(in_chan, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    return _make_efficientnet_backbone(efficientnet)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape2 = out_shape3 = out_shape
    out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, in_chan=3, group_width=8):
    if backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained, in_chan=in_chan, group_width=group_width)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)
    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable, in_chan=in_chan)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)
    else:
        raise ValueError(f"Backbone '{backbone}' not implemented")
    return pretrained, scratch


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features // 2 if self.expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class MidasNet(BaseModel):
    def __init__(self, pretrained=False, features=256, input_channels=3, output_channels=1, group_width=8, last_residual=False):
        super().__init__()
        self.out_chan = output_channels
        self.last_res = last_residual
        self.pretrained, self.scratch = _make_encoder(
            backbone="resnext101_wsl",
            features=features,
            use_pretrained=pretrained,
            in_chan=input_channels,
            group_width=group_width,
        )
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        out_act = nn.Sigmoid()

        res_dim = 128 + (input_channels if last_residual else 0)
        self.scratch.output_conv = nn.ModuleList([
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(res_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            out_act,
        ])

    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out)
        if self.last_res:
            out = torch.cat((out, x), dim=1)
        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)
        return out


class MidasNet_small(BaseModel):
    def __init__(
            self,
            pretrained=False,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            channels_last=False,
            align_corners=True,
            blocks={"expand": True},
            input_channels=3,
            output_channels=1,
            out_bias=0,
            last_residual=False,
    ):
        super().__init__()
        self.out_chan = output_channels
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone
        self.last_res = last_residual
        self.groups = 1

        features1 = features2 = features3 = features4 = features
        self.expand = False
        if self.blocks.get("expand"):
            self.expand = True
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, pretrained, in_chan=input_channels, groups=self.groups, expand=self.expand, exportable=exportable)
        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, expand=False, align_corners=align_corners)

        output_act = nn.Sigmoid()

        res_dim = (features // 2) + (input_channels if last_residual else 0)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(res_dim, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            output_act,
        )
        self.scratch.output_conv[-2].bias = torch.nn.Parameter(torch.ones(output_channels) * out_bias)

    def forward(self, x):
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out)
        if self.last_res:
            out = torch.cat((out, x), dim=1)
        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)
        return out


def round_32(x):
    return 32 * math.ceil(x / 32)


def invert(x):
    return 1.0 / (x + 1.0)


def uninvert(x, eps=0.001, clip=True):
    if clip:
        x = x.clip(eps, 1.0)
    return (1.0 / x) - 1.0


def get_brightness(rgb, mode="numpy", keep_dim=True):
    if mode == "torch" or torch.is_tensor(rgb):
        brightness = (0.3 * rgb[0, :, :]) + (0.59 * rgb[1, :, :]) + (0.11 * rgb[2, :, :])
        if keep_dim:
            brightness = brightness.unsqueeze(0)
        return brightness
    brightness = (0.3 * rgb[:, :, 0]) + (0.59 * rgb[:, :, 1]) + (0.11 * rgb[:, :, 2])
    if keep_dim:
        brightness = brightness[:, :, np.newaxis]
    return brightness


def batch_rgb2iuv(rgb, eps=0.001):
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]
    l = (r * 0.299) + (g * 0.587) + (b * 0.114)
    i = invert(l)
    u = invert(r / (g + eps))
    v = invert(b / (g + eps))
    return torch.stack((i, u, v), dim=1)


def batch_iuv2rgb(iuv, eps=0.001):
    l = uninvert(iuv[:, 0, :, :], eps=eps)
    u = uninvert(iuv[:, 1, :, :], eps=eps)
    v = uninvert(iuv[:, 2, :, :], eps=eps)
    g = l / ((u * 0.299) + (v * 0.114) + 0.587)
    r = g * u
    b = g * v
    return torch.stack((r, g, b), dim=1)


def optimal_resize(img, conf=0.01):
    if conf is None or conf <= 0:
        h, w = img.shape[:2]
        return resize(img, (round_32(h), round_32(w)), anti_aliasing=True)
    h, w, _ = img.shape
    max_dim = max(h, w)
    target = min(max_dim * (1.0 + conf), 1500)
    scale = target / max_dim
    return resize(img, (round_32(h * scale), round_32(w * scale)), anti_aliasing=True)


def load_decompile(path):
    compiled_dict = torch.load(path)
    remove_prefix = "_orig_mod."
    return {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in compiled_dict.items()}


def load_models(device="cuda"):
    models = {}

    base_url = "https://github.com/compphoto/Intrinsic/releases/download/v2.1/"
    ord_state_dict = torch.hub.load_state_dict_from_url(base_url + "stage_0_v21.pt", map_location=device, progress=True)
    iid_state_dict = torch.hub.load_state_dict_from_url(base_url + "stage_1_v21.pt", map_location=device, progress=True)
    col_state_dict = torch.hub.load_state_dict_from_url(base_url + "stage_2_v21.pt", map_location=device, progress=True)
    alb_state_dict = torch.hub.load_state_dict_from_url(base_url + "stage_3_v21.pt", map_location=device, progress=True)
    alb_residual = True

    ord_model = MidasNet()
    ord_model.load_state_dict(ord_state_dict)
    ord_model.eval()
    models["ord_model"] = ord_model.to(device)

    iid_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    iid_model.load_state_dict(iid_state_dict)
    iid_model.eval()
    models["iid_model"] = iid_model.to(device)

    col_model = MidasNet(input_channels=7, output_channels=2)
    col_model.load_state_dict(col_state_dict)
    col_model.eval()
    models["col_model"] = col_model.to(device)

    alb_model = MidasNet(input_channels=9, output_channels=3, last_residual=alb_residual)
    alb_model.load_state_dict(alb_state_dict)
    alb_model.eval()
    models["alb_model"] = alb_model.to(device)

    return models


def base_resize(img, base_size=384):
    h, w, _ = img.shape
    max_dim = max(h, w)
    scale = base_size / max_dim
    new_h, new_w = round_32(h * scale), round_32(w * scale)
    return resize(img, (new_h, new_w, 3), anti_aliasing=True)


def equalize_predictions(img, base, full, p=0.5):
    h, w, _ = img.shape
    full_shd = (1.0 / full.clip(1e-5)) - 1.0
    base_shd = (1.0 / base.clip(1e-5)) - 1.0
    full_alb = get_brightness(img) / full_shd.clip(1e-5)
    base_alb = get_brightness(img) / base_shd.clip(1e-5)
    rand_msk = (np.random.randn(h, w) > p).astype(np.uint8)
    flat_full_alb = full_alb[rand_msk == 1]
    flat_base_alb = base_alb[rand_msk == 1]
    scale, _, _, _ = np.linalg.lstsq(flat_full_alb.reshape(-1, 1), flat_base_alb, rcond=None)
    new_full_alb = scale * full_alb
    new_full_shd = get_brightness(img) / new_full_alb.clip(1e-5)
    new_full = 1.0 / (1.0 + new_full_shd)
    return base, new_full


def run_gray_pipeline(
        models,
        img_arr,
        base_size=384,
        device="cuda",
        lstsq_p=0.0,
):
    orig_h, orig_w, _ = img_arr.shape

    img_arr = resize(img_arr, (round_32(orig_h), round_32(orig_w)), anti_aliasing=True)

    fh, fw, _ = img_arr.shape
    lin_img = img_arr ** 2.2

    with torch.no_grad():
        base_input = base_resize(lin_img, base_size)
        full_input = lin_img

        base_input = torch.from_numpy(base_input).permute(2, 0, 1).to(device).float()
        full_input = torch.from_numpy(full_input).permute(2, 0, 1).to(device).float()

        base_out = models["ord_model"](base_input.unsqueeze(0)).squeeze(0)
        full_out = models["ord_model"](full_input.unsqueeze(0)).squeeze(0)

        base_out = base_out.permute(1, 2, 0).cpu().numpy()
        full_out = full_out.permute(1, 2, 0).cpu().numpy()
        base_out = resize(base_out, (fh, fw))

        ord_base, ord_full = equalize_predictions(lin_img, base_out, full_out, p=lstsq_p)

        inp = torch.from_numpy(lin_img).permute(2, 0, 1).to(device)
        bse = torch.from_numpy(ord_base).permute(2, 0, 1).to(device)
        fll = torch.from_numpy(ord_full).permute(2, 0, 1).to(device)

        combined = torch.cat((inp, bse, fll), 0).unsqueeze(0)

        inv_shd = models["iid_model"](combined).squeeze(1)
        shd = uninvert(inv_shd)
        alb = inp / shd

    inv_shd = inv_shd.squeeze(0).detach().cpu().numpy()
    alb = alb.permute(1, 2, 0).detach().cpu().numpy()

    return {
        "lin_img": lin_img,
        "gry_shd": inv_shd,
        "gry_alb": alb,
    }


def run_pipeline(models, img_arr, base_size=384, device="cuda"):
    results = run_gray_pipeline(
        models,
        img_arr,
        device=device,
        base_size=base_size,
    )

    img = results["lin_img"]
    gry_shd = results["gry_shd"][:, :, None]
    gry_alb = results["gry_alb"]

    net_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    net_shd = torch.from_numpy(gry_shd).permute(2, 0, 1).unsqueeze(0).to(device)
    net_alb = torch.from_numpy(gry_alb).permute(2, 0, 1).unsqueeze(0).to(device)

    in_img_luv = batch_rgb2iuv(net_img)
    in_alb_luv = batch_rgb2iuv(net_alb)

    orig_sz = img.shape[:2]
    scale = base_size / max(orig_sz)
    base_sz = (round_32(orig_sz[0] * scale), round_32(orig_sz[1] * scale))

    in_img_luv = torch.nn.functional.interpolate(in_img_luv, size=base_sz, mode="bilinear", align_corners=True, antialias=True)
    in_alb_luv = torch.nn.functional.interpolate(in_alb_luv, size=base_sz, mode="bilinear", align_corners=True, antialias=True)
    in_gry_shd = torch.nn.functional.interpolate(net_shd, size=base_sz, mode="bilinear", align_corners=True, antialias=True)

    inp = torch.cat([in_img_luv, in_gry_shd, in_alb_luv], 1)

    with torch.no_grad():
        uv_shd = models["col_model"](inp)

    uv_shd = torch.nn.functional.interpolate(uv_shd, size=orig_sz, mode="bilinear", align_corners=True)

    iuv_shd = torch.cat((net_shd, uv_shd), 1)
    rough_shd = batch_iuv2rgb(iuv_shd)
    rough_alb = net_img / rough_shd

    rough_alb *= 0.75 / torch.quantile(rough_alb, 0.99)
    rough_alb = rough_alb.clip(0.001)
    rough_shd = net_img / rough_alb

    inp = torch.cat([net_img, invert(rough_shd), rough_alb], 1)
    with torch.no_grad():
        pred_alb = models["alb_model"](inp)

    hr_alb = pred_alb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return {"hr_alb": hr_alb}

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_hr_alb_image(hr_alb, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = hr_alb ** (1 / 2.2)
    max_val = float(img.max()) if img.size else 1.0
    denom = max(max_val, 1e-6)
    img = ((img / denom).clip(0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)


def iter_images(path):
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in suffixes])


def main():
    parser = argparse.ArgumentParser(description="Predict hr_alb using the intrinsic pipeline.")
    parser.add_argument("--input", default="inputs", help="Image file or folder of images.")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: <input>/albedo_pred).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device=device)

    input_path = Path(args.input)
    if input_path.is_dir():
        output_dir = Path(args.output_dir) if args.output_dir else input_path / "albedo_pred"
        for img_path in iter_images(input_path):
            img = load_image(str(img_path))
            result = run_pipeline(models, img, device=device)
            save_hr_alb_image(result["hr_alb"], output_dir / img_path.name)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "albedo_pred"
        img = load_image(str(input_path))
        result = run_pipeline(models, img, device=device)
        save_hr_alb_image(result["hr_alb"], output_dir / input_path.name)


if __name__ == "__main__":
    main()
