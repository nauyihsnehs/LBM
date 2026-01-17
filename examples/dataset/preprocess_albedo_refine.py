import logging
from pathlib import Path
from typing import Iterable, Optional

import fire
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

from lbm.intrinsic import AlbedoInference

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _list_rgb_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in VALID_SUFFIXES
            and path.stem.endswith("_rgb")
        ):
            yield path


def _load_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transforms.ToTensor()(image)


def _robedo_path(
    source_path: Path,
    output_root: Optional[Path],
    robedo_suffix: str,
    robedo_extension: str,
) -> Path:
    robedo_stem = source_path.stem[: -len("_rgb")] + robedo_suffix
    robedo_name = f"{robedo_stem}{robedo_extension}"
    if output_root is None:
        return source_path.with_name(robedo_name)
    relative = source_path.parent.name
    output_dir = output_root / relative
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / robedo_name


def main(
    data_root: str,
    output_root: Optional[str] = None,
    device: Optional[str] = None,
    base_size: int = 512,
    robedo_suffix: str = "_rlb",
    robedo_extension: str = ".png",
):
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    output_root_path = Path(output_root) if output_root else None

    model = AlbedoInference(device=device, base_size=base_size)
    to_pil = ToPILImage()

    person_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not person_dirs:
        raise ValueError(f"No subdirectories found under {root}")

    processed = 0
    for person_dir in sorted(person_dirs):
        for rgb_path in _list_rgb_images(person_dir):
            robedo_path = _robedo_path(
                rgb_path,
                output_root_path,
                robedo_suffix,
                robedo_extension,
            )
            if robedo_path.exists():
                continue

            source_tensor = _load_tensor(rgb_path)
            with torch.inference_mode():
                robedo_tensor = model(source_tensor.unsqueeze(0)).squeeze(0).cpu()
            robedo_tensor = robedo_tensor.clamp(0.0, 1.0)
            robedo_image = to_pil(robedo_tensor)
            robedo_image.save(robedo_path)
            processed += 1
            print(f'{person_dir} -> {rgb_path} -> {robedo_path}')

    logger.info("Generated %d rough albedo images.", processed)


if __name__ == "__main__":
    fire.Fire(main)
