import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

from lbm.moge import DepthBatchInferencer


def _iter_rgb_images(root: Path) -> Iterable[Path]:
    return sorted(root.glob("*/*_999_rgb.png"))


def _parse_ids(path: Path) -> Tuple[int, int, int, int]:
    scene_dir = path.parent.name
    pos_light = path.stem
    scene_id, human_id = (int(part) for part in scene_dir.split("_", maxsplit=1))
    pos_id, light_id, _ = pos_light.split("_", maxsplit=2)
    return scene_id, human_id, int(pos_id), int(light_id)


def _output_path(output_root: Path, path: Path) -> Path:
    scene_id, human_id, pos_id, light_id = _parse_ids(path)
    scene_dir = f"{scene_id:03d}_{human_id:03d}"
    filename = f"{pos_id:03d}_dpt.exr"
    return output_root / scene_dir / filename


def _load_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image


def _save_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = depth.astype(np.float32)
    cv2.imwrite(
        str(path),
        depth,
        [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_ZIP,
        ],
    )


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    device = torch.device(args.device)

    inferencer = DepthBatchInferencer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        device=device,
        use_fp16=args.use_fp16,
    )

    images = _iter_rgb_images(input_root)
    batch: List[torch.Tensor] = []
    batch_paths: List[Path] = []

    for image_path in images:
        output_path = _output_path(output_root, image_path)
        if output_path.exists():
            continue

        batch.append(_load_image(image_path))
        batch_paths.append(image_path)

        if len(batch) < args.batch_size:
            continue

        _run_batch(inferencer, batch, batch_paths, output_root, args)
        batch = []
        batch_paths = []

    if batch:
        _run_batch(inferencer, batch, batch_paths, output_root, args)


def _run_batch(
        inferencer: DepthBatchInferencer,
        batch: List[torch.Tensor],
        batch_paths: List[Path],
        output_root: Path,
        args: argparse.Namespace,
) -> None:
    images = torch.stack(batch, dim=0).to(device=inferencer.device)
    outputs = inferencer.infer_batch(
        images,
        fov_x=args.fov_x,
        resolution_level=args.resolution_level,
        num_tokens=args.num_tokens,
        use_fp16=args.use_fp16,
    )
    depth_batch = outputs["depth"].detach().cpu().numpy()
    # turn infinite depths to zeros
    depth_batch[np.isinf(depth_batch)] = 0.0
    # turn NaN depths to zeros
    depth_batch[np.isnan(depth_batch)] = 0.0
    for depth, input_path in zip(depth_batch, batch_paths):
        output_path = _output_path(output_root, input_path)
        if output_path.exists():
            continue
        _save_depth(output_path, depth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for MoGe depth.")
    parser.add_argument("--model", required=False, default="Ruicheng/moge-2-vitl")
    # parser.add_argument("--input-root", required=False, default="/mnt/data1/ssy/render_people/fill-light-dataset/train")
    # parser.add_argument("--output-root", required=False, default="/mnt/data1/ssy/render_people/fill-light-dataset/train")
    parser.add_argument("--input-root", required=False, default="/mnt/data1/ssy/render_people/fill-light-dataset/val")
    parser.add_argument("--output-root", required=False, default="/mnt/data1/ssy/render_people/fill-light-dataset/val")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda/cpu).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--use-fp16", action="store_true", help="Enable FP16 inference.")
    parser.add_argument("--resolution-level", type=int, default=9, help="Inference resolution level.")
    parser.add_argument("--num-tokens", type=int, default=None, help="Override number of tokens.")
    parser.add_argument("--fov-x", type=float, default=None, help="Optional horizontal FoV in degrees.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
