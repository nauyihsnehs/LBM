import argparse
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
from PIL import Image

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPaths:
    base_dir: Path
    rgb_path: Path
    target_path: Path
    lighting_path: Path


def _iter_images(input_root: Path, patterns: Optional[Sequence[str]]) -> Iterable[Path]:
    if patterns:
        for pattern in patterns:
            yield from sorted(input_root.rglob(pattern))
        return
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def _scene_pos_from_index(index: int, group_size: int) -> tuple[int, int]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return index // group_size, index % group_size


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_or_convert(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    _ensure_parent(destination)
    if source.suffix.lower() == ".png":
        shutil.copy2(source, destination)
        return
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.save(destination, format="PNG")


def _write_black_lighting_params(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    _ensure_parent(path)
    theta_norm = random.random()
    phi_norm = random.random()
    color = (random.random(), random.random(), random.random())
    params = (
        float(theta_norm),
        float(phi_norm),
        0.0,
        float(color[0]),
        float(color[1]),
        float(color[2]),
        0.0,
    )
    with h5py.File(path, "w") as file:
        file.create_dataset("lighting_params", data=list(params))


def _build_paths(
        output_root: Path,
        index: int,
        group_size: int,
        light_id: str,
) -> DatasetPaths:
    scene_id, pos_index = _scene_pos_from_index(index, group_size)
    human_id = 0
    pos_id = f"{pos_index:03d}"
    base_dir = output_root / f"{scene_id:03d}_{human_id:03d}"
    rgb_path = base_dir / f"{pos_id}_999_rgb.png"
    target_path = base_dir / f"{pos_id}_{light_id}_rgb.png"
    lighting_path = base_dir / f"{pos_id}_{light_id}_lgt.h5"
    return DatasetPaths(
        base_dir=base_dir,
        rgb_path=rgb_path,
        target_path=target_path,
        lighting_path=lighting_path,
    )


def _build_dataset(
        input_root: Path,
        output_root: Path,
        group_size: int,
        overwrite: bool,
        patterns: Optional[Sequence[str]],
        light_id: str,
) -> None:
    images = list(_iter_images(input_root, patterns))
    if not images:
        raise FileNotFoundError(f"No images found in {input_root}")

    for index, image_path in enumerate(images):
        paths = _build_paths(
            output_root,
            index,
            group_size,
            light_id,
        )
        _copy_or_convert(image_path, paths.rgb_path, overwrite)
        _ensure_parent(paths.target_path)
        if not paths.target_path.exists() or overwrite:
            shutil.copy2(paths.rgb_path, paths.target_path)
        _write_black_lighting_params(paths.lighting_path, overwrite=overwrite)


def _run_subprocess(command: list[str]) -> None:
    logger.info("Running: %s", " ".join(command))
    result = os.spawnv(os.P_WAIT, command[0], command)
    if result != 0:
        raise RuntimeError(f"Command failed with exit code {result}: {' '.join(command)}")


def _run_albedo_estimation(
        output_root: Path,
        albedo_suffix: str,
        albedo_extension: str,
        overwrite: bool,
        extra_args: Sequence[str],
) -> None:
    command = [
        shutil.which("python") or "python",
        str(Path(__file__).with_name("preprocess_albedo_estimation.py")),
        f"--data_root={output_root}",
        f"--albedo_suffix={albedo_suffix}",
        f"--albedo_extension={albedo_extension}",
        f"--overwrite={str(overwrite)}",
        *extra_args,
    ]
    _run_subprocess(command)


def _run_depth_estimation(
        output_root: Path,
        extra_args: Sequence[str],
) -> None:
    command = [
        shutil.which("python") or "python",
        str(Path(__file__).with_name("preprocess_depth.py")),
        "--input-root",
        str(output_root),
        "--output-root",
        str(output_root),
        *extra_args,
    ]
    _run_subprocess(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fake fill-lighting dataset for self-supervision.",
    )
    parser.add_argument("--input-root", type=Path, default='/mnt/data1/ssy/render_people/fill-light-dataset/train-hhfq')
    parser.add_argument("--output-root", type=Path,
                        default='/mnt/data1/ssy/render_people/fill-light-dataset/train-self')
    parser.add_argument(
        "--group-size",
        type=int,
        default=100,
        help="Number of frames per scene id before incrementing (default: 100).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--light-id", default="000", help="3-digit light id (<100).")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Optional glob pattern(s) relative to input-root.",
    )
    parser.add_argument(
        "--run-albedo",
        action="store_true",
        help="Run preprocess_albedo_estimation.py after building the dataset.",
    )
    parser.add_argument(
        "--run-depth",
        action="store_true",
        help="Run preprocess_depth.py after building the dataset.",
    )
    parser.add_argument("--albedo-suffix", default="_alb")
    parser.add_argument("--albedo-extension", default=".png")
    parser.add_argument(
        "--albedo-extra-args",
        nargs=argparse.REMAINDER,
        default=(),
        help="Extra args passed to preprocess_albedo_estimation.py.",
    )
    parser.add_argument(
        "--depth-extra-args",
        nargs=argparse.REMAINDER,
        default=(),
        help="Extra args passed to preprocess_depth.py.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    _build_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        group_size=args.group_size,
        overwrite=args.overwrite,
        patterns=args.pattern,
        light_id=args.light_id,
    )
    if args.run_albedo:
        _run_albedo_estimation(
            output_root=args.output_root,
            albedo_suffix=args.albedo_suffix,
            albedo_extension=args.albedo_extension,
            overwrite=args.overwrite,
            extra_args=args.albedo_extra_args,
        )
    if args.run_depth:
        _run_depth_estimation(
            output_root=args.output_root,
            extra_args=args.depth_extra_args,
        )


if __name__ == "__main__":
    main()
