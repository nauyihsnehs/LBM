import io
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import fire
import webdataset as wds
from PIL import Image


def _load_and_resize(path: Path, size: int) -> bytes:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _iter_samples(
    dataset_root: Path,
    image_size: int,
) -> Iterable[Tuple[str, Dict[str, bytes]]]:
    people_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    filename_re = re.compile(r"^(\d{4})_(\d{2})\.png$")

    for person_dir in people_dirs:
        person_id = person_dir.name
        rgb_dir = person_dir / "rgb"
        fill_dir = person_dir / "fill"
        shading_dir = person_dir / "diffuse_shading"
        normal_dir = person_dir / "normal"

        if not (rgb_dir.exists() and fill_dir.exists() and shading_dir.exists()):
            logging.warning("Skipping %s due to missing subdirectories", person_dir)
            continue

        for fill_path in sorted(fill_dir.glob("*.png")):
            match = filename_re.match(fill_path.name)
            if match is None:
                logging.warning("Skipping unexpected fill filename: %s", fill_path)
                continue

            frame_id = int(match.group(1))
            fill_id = int(match.group(2))

            rgb_path = rgb_dir / f"{frame_id:05d}.png"
            shading_path = shading_dir / f"{frame_id:04d}_{fill_id:02d}.png"
            normal_path = normal_dir / f"{frame_id:04d}.png"
            target_path = fill_path

            if not (rgb_path.exists() and shading_path.exists() and normal_path.exists()):
                logging.warning(
                    "Missing files for person %s frame %04d fill %02d", person_id, frame_id, fill_id
                )
                continue

            sample_key = f"{person_id}_{frame_id:04d}_{fill_id:02d}"
            sample = {
                "source.png": _load_and_resize(rgb_path, image_size),
                "target.png": _load_and_resize(target_path, image_size),
                "shading.png": _load_and_resize(shading_path, image_size),
                "normal.png": _load_and_resize(normal_path, image_size),
            }
            yield sample_key, sample


def build_webdataset(
    dataset_root: str = "/mnt/data1/ssy/render_people/render",
    output_dir: str = "./relight_webdataset",
    shard_size: int = 1000,
    image_size: int = 512,
    max_samples: Optional[int] = None,
) -> None:
    dataset_root_path = Path(dataset_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_pattern = str(output_path / "relight-%06d.tar")
    sample_count = 0

    with wds.ShardWriter(shard_pattern, maxcount=shard_size) as sink:
        for key, sample in _iter_samples(dataset_root_path, image_size):
            sink.write({"__key__": key, **sample})
            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break

    logging.info("Wrote %d samples to %s", sample_count, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(build_webdataset)
