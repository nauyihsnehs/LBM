#!/usr/bin/env python3
"""Filter {pos}_999_rgb.png images by luminance thresholds.

Moves images with too many near-white or near-black pixels into a recycle
directory, preserving relative paths from the input root.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

RGB_SUFFIX = "_999_rgb.png"


def _iter_rgb_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob(f"*{RGB_SUFFIX}") if path.is_file())


def _mean_intensity(rgb: np.ndarray) -> np.ndarray:
    rgb_float = rgb.astype(np.float32) / 255.0
    return rgb_float.mean(axis=-1)


def _should_recycle(lum: np.ndarray, threshold_ratio: float) -> tuple[bool, float, float]:
    total = lum.size
    white_ratio = float(np.count_nonzero(lum >= 0.99)) / total
    black_ratio = float(np.count_nonzero(lum <= 0.01)) / total
    return white_ratio > threshold_ratio or black_ratio > threshold_ratio, white_ratio, black_ratio


def _move_to_recycle(src: Path, recycle_root: Path, input_root: Path, dry_run: bool) -> Path:
    rel_path = src.relative_to(input_root)
    dest = recycle_root / rel_path
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter rgb images with extreme luminance.")
    parser.add_argument("--input-root", type=Path,
                        default="/mnt/data1/ssy/render_people/fill-light-dataset/fill")
    parser.add_argument("--recycle-root", type=Path,
                        default="/mnt/data1/ssy/render_people/fill-light-dataset/_trash_bin")
    parser.add_argument("--threshold-ratio", type=float, default=1 / 3)
    parser.add_argument("--dry-run", default=False, help="Print actions without moving files.")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    recycle_root = args.recycle_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    rgb_paths = _iter_rgb_images(input_root)
    if not rgb_paths:
        print(f"No matching images found under {input_root}")
        return

    filtered = 0
    for rgb_path in rgb_paths:
        try:
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:
            print(f"[ERR] {rgb_path}: {exc}")
            continue

        mean_intensity = _mean_intensity(rgb)
        should_recycle, white_ratio, black_ratio = _should_recycle(
            mean_intensity, args.threshold_ratio
        )
        if should_recycle:
            dest = _move_to_recycle(rgb_path, recycle_root, input_root, args.dry_run)
            filtered += 1
            status = "[DRY]" if args.dry_run else "[MOVE]"
            print(
                f"{status} {rgb_path} -> {dest} "
                f"(white={white_ratio:.3f}, black={black_ratio:.3f})"
            )

    print(f"Filtered {filtered} images out of {len(rgb_paths)} total.")


if __name__ == "__main__":
    main()
