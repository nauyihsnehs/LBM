import torch

from inference_albedo_single_file import load_models, run_pipeline


class AlbedoInference(torch.nn.Module):
    def __init__(self, device=None, base_size=384):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.base_size = base_size
        self.models = torch.nn.ModuleDict(load_models(device=self.device))
        self.eval()

    def forward(self, batch_tensor):
        batch_tensor = batch_tensor.to(self.device, non_blocking=True)
        outputs = []
        with torch.inference_mode():
            for item in batch_tensor:
                img = item.detach().permute(1, 2, 0).cpu().numpy()
                pred = run_pipeline(self.models, img, base_size=self.base_size, device=self.device)["hr_alb"]
                outputs.append(torch.from_numpy(pred).permute(2, 0, 1).to(self.device))
        return torch.stack(outputs, dim=0)
