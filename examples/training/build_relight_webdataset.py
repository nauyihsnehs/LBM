import io
import logging
import re
import time
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
    start_after: Optional[Tuple[str, int, int]] = None,
) -> Iterable[Tuple[str, Dict[str, bytes], str, int, int]]:
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

            key_tuple = (person_id, frame_id, fill_id)
            if start_after is not None and key_tuple <= start_after:
                continue

            sample_key = f"{person_id}_{frame_id:04d}_{fill_id:02d}"
            sample = {
                "source.png": _load_and_resize(rgb_path, image_size),
                "target.png": _load_and_resize(target_path, image_size),
                "shading.png": _load_and_resize(shading_path, image_size),
                "normal.png": _load_and_resize(normal_path, image_size),
            }
            yield sample_key, sample, person_id, frame_id, fill_id


def _parse_sample_key(sample_key: str) -> Optional[Tuple[str, int, int]]:
    match = re.match(r"^(.+?)_(\d{4})_(\d{2})$", sample_key)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _existing_output_state(
    output_path: Path,
) -> Tuple[int, Optional[Tuple[str, int, int]], int]:
    shard_paths = sorted(output_path.glob("relight-*.tar"))
    if not shard_paths:
        return 0, None, 0

    existing_count = 0
    last_key: Optional[Tuple[str, int, int]] = None
    dataset = wds.WebDataset(str(output_path / "relight-*.tar"))
    for sample in dataset:
        key = sample.get("__key__")
        if key is None:
            continue
        parsed = _parse_sample_key(key)
        if parsed is None:
            continue
        existing_count += 1
        last_key = parsed

    shard_indices = []
    for shard_path in shard_paths:
        match = re.search(r"relight-(\d+)\.tar$", shard_path.name)
        if match:
            shard_indices.append(int(match.group(1)))

    next_shard = max(shard_indices) + 1 if shard_indices else 0
    return existing_count, last_key, next_shard


def build_webdataset(
    dataset_root: str = "/mnt/data1/ssy/render_people/render",
    output_dir: str = "./relight_webdataset",
    shard_size: int = 1000,
    image_size: int = 512,
    max_samples: Optional[int] = None,
    log_interval: int = 1000,
    resume: bool = True,
) -> None:
    dataset_root_path = Path(dataset_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_pattern = str(output_path / "relight-%06d.tar")
    sample_count = 0
    start_after: Optional[Tuple[str, int, int]] = None
    start_shard = 0

    if resume:
        existing_count, last_key, start_shard = _existing_output_state(output_path)
        sample_count = existing_count
        start_after = last_key
        if existing_count:
            logging.info(
                "Resuming from %d existing samples; last key=%s", existing_count, last_key
            )

    processed_count = 0
    start_time = time.monotonic()
    last_ids: Optional[Tuple[str, int, int]] = None

    with wds.ShardWriter(
        shard_pattern,
        maxcount=shard_size,
        start_shard=start_shard,
    ) as sink:
        for key, sample, person_id, frame_id, fill_id in _iter_samples(
            dataset_root_path,
            image_size,
            start_after=start_after,
        ):
            sink.write({"__key__": key, **sample})
            sample_count += 1
            processed_count += 1
            last_ids = (person_id, frame_id, fill_id)
            if log_interval and processed_count % log_interval == 0:
                elapsed = max(time.monotonic() - start_time, 1e-6)
                samples_per_sec = processed_count / elapsed
                logging.info(
                    "Processed %d samples (total %d). %.2f samples/sec. Last=%s frame=%04d fill=%02d",
                    processed_count,
                    sample_count,
                    samples_per_sec,
                    person_id,
                    frame_id,
                    fill_id,
                )
            if max_samples is not None and processed_count >= max_samples:
                break

    if processed_count and last_ids:
        elapsed = max(time.monotonic() - start_time, 1e-6)
        samples_per_sec = processed_count / elapsed
        logging.info(
            "Finished %d samples in %.2fs (%.2f samples/sec). Last=%s frame=%04d fill=%02d",
            processed_count,
            elapsed,
            samples_per_sec,
            last_ids[0],
            last_ids[1],
            last_ids[2],
        )
    logging.info("Wrote %d total samples to %s", sample_count, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(build_webdataset)
