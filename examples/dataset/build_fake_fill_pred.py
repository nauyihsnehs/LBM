# build_fake_fill_pred.py
import argparse
import re
from pathlib import Path

from PIL import Image

RGB_RE = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_rgb\.png$", re.IGNORECASE)


def process_folder(src_root: Path, dry_run: bool = False) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    total = 0
    written = 0
    skipped = 0

    # Walk only files ending with _rgb.png
    for rgb_path in src_root.rglob("*_rgb.png"):
        if not rgb_path.is_file():
            continue

        m = RGB_RE.match(rgb_path.name)
        if not m:
            # filename doesn't match {pos:03d}_{light:03d}_rgb.png
            skipped += 1
            continue

        pos = m.group("pos")                 # already 3 digits
        light_str = m.group("light")         # already 3 digits
        light_id = int(light_str)

        if light_id >= 100:
            skipped += 1
            continue

        out_name = f"{pos}_{light_str}_pre.png"
        out_path = rgb_path.with_name(out_name)

        total += 1

        if dry_run:
            print(f"[DRY] {rgb_path} -> {out_path}")
            continue
        else:
            print(f"[DRY] {rgb_path} -> {out_path}")

        try:
            with Image.open(rgb_path) as im:
                w, h = im.size
                new_w, new_h = w // 4, h // 4
                if new_w < 1 or new_h < 1:
                    print(f"[SKIP] Too small to downsample: {rgb_path} ({w}x{h})")
                    skipped += 1
                    continue

                im2 = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

                # Ensure output directory exists (should, but safe)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                im2.save(out_path, format="PNG", optimize=True)

            written += 1
            if written % 200 == 0:
                print(f"[OK] written {written}... (latest: {out_path})")

        except Exception as e:
            print(f"[ERR] {rgb_path}: {e}")
            skipped += 1

    print("\nDone.")
    print(f"Matched candidates: {total}")
    print(f"Written:          {written}")
    print(f"Skipped:          {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        # default=r"E:\evermotion\train-set",
        default=r"/mnt/data1/ssy/render_people/fill-light-dataset/train/render",
        help="Root folder containing {scene_id:03d}_{human_id:03d}/...",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    args = ap.parse_args()

    process_folder(Path(args.src), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
