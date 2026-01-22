import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py

logger = logging.getLogger(__name__)

FILENAME_PATTERN = re.compile(r"(?P<pos>\d{3})_(?P<light>\d{3})_rgb")


def _extract_ids(frame: Dict) -> Optional[Tuple[str, str]]:
    pos_id = frame.get("pos_id")
    light_id = frame.get("light_id")
    if pos_id is not None and light_id is not None:
        return f"{int(pos_id):03d}", f"{int(light_id):03d}"

    for value in frame.values():
        if isinstance(value, str):
            match = FILENAME_PATTERN.search(value)
            if match:
                return match.group("pos"), match.group("light")

    return None


def _lighting_params_from_frame(frame: Dict, relight=False) -> Optional[Tuple[str, Iterable[float]]]:
    ids = _extract_ids(frame)
    if ids is None:
        return None
    pos_id, light_id = ids
    if int(light_id) == 999:
        return None
    light = frame.get("light")
    if light is None:
        raise KeyError(f"Missing light data for frame {pos_id}_{light_id}.")

    theta_deg = float(light["theta_deg"])
    phi_deg = float(light["phi_deg"])
    theta_norm = theta_deg / 180.0
    phi_norm = phi_deg / 90.0
    power_cfg = float(light["power_cfg"])
    color = light["color"]
    if len(color) != 3:
        raise ValueError(f"Expected RGB color for frame {pos_id}_{light_id}.")
    color_r, color_g, color_b = [float(c) for c in color]
    distance = float(light["distance_D_m"])
    radius = float(light["radius_R_m"])
    constrain_radius = math.atan(radius / distance) if distance != 0 else 0.0
    params = (
        theta_norm,
        phi_norm,
        power_cfg,
        color_r,
        color_g,
        color_b,
        constrain_radius
    )
    # print(theta_norm, phi_norm)
    # get original_lighting_scale if relight
    if relight:
        original_lighting_scale = float(light.get("original_lighting_scale", 1.0))
        params += (original_lighting_scale, )
    return f"{pos_id}_{light_id}", params


def _load_metadata(metadata_path: Path) -> Dict:
    with metadata_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    renders = data.get("renders")
    if renders is None:
        raise KeyError(f"Missing 'renders' in metadata: {metadata_path}")
    return data


def generate_h5_files(root_dir: Path, overwrite: bool = False, relight=False) -> None:
    base_dirs = [path for path in root_dir.iterdir() if path.is_dir()]
    if not base_dirs:
        raise FileNotFoundError(f"No base directories found in {root_dir}")

    for base_dir in sorted(base_dirs):
        if relight:
            metadata_path = base_dir / "metadata-relight.json"
        else:
            metadata_path = base_dir / "metadata.json"
        if not metadata_path.exists():
            logger.warning("Skipping %s (missing metadata.json).", base_dir)
            continue
        data = _load_metadata(metadata_path)
        renders = data["renders"]
        for frame in renders:
            result = _lighting_params_from_frame(frame, relight=relight)
            if result is None:
                continue
            stem, params = result
            output_path = base_dir / f"{stem}_lgt.h5"
            if output_path.exists() and not overwrite:
                logger.info("Skipping existing %s", output_path)
                continue
            with h5py.File(output_path, "w") as file:
                file.create_dataset("lighting_params", data=list(params))
            logger.info("Wrote %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lighting parameter .h5 files from metadata.json."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        # default='/mnt/data1/ssy/render_people/fill-light-dataset/train',
        # default='/mnt/data1/ssy/render_people/fill-light-dataset/val',
        # default='/mnt/data1/ssy/render_people/fill-light-dataset/test',
        default='/mnt/data1/ssy/render_people/fill-light-dataset/train-re',
        help="Root directory containing base scene/human folders.",
    )
    parser.add_argument(
        "--relight",
        default=True,
        help="Root directory containing base scene/human folders.",
    )
    parser.add_argument(
        "--overwrite",
        default=True,
        help="Overwrite existing .h5 files if present.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    generate_h5_files(args.root_dir, overwrite=args.overwrite, relight=args.relight)


if __name__ == "__main__":
    main()
