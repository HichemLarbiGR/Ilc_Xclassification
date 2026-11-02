import os
import sys
from pathlib import Path
import openslide
from PIL import Image

# Requested scale factor
SCALE_FACTOR = 16


def get_downsample_level(slide, target_factor=16):
    """
    Return the OpenSlide level whose downsample factor is closest to
    (without exceeding) the target factor.
    If no native level is low enough, we fall back to the lowest level
    and resize afterwards.
    """
    best_level = 0
    best_factor = 1.0
    for level, factor in enumerate(slide.level_downsamples):
        if factor <= target_factor and factor > best_factor:
            best_level = level
            best_factor = factor
    return best_level, best_factor


def wsi_to_png(wsi_path, output_dir):
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        print(f"[ERROR] Cannot open {wsi_path}: {e}")
        return

    try:
        level, native_factor = get_downsample_level(slide, SCALE_FACTOR)
        level_dims = slide.level_dimensions[level]  # (width, height) at chosen level

        # Read the whole image at the selected level
        # read_region expects (x, y) in level-0 coordinates
        region = slide.read_region((0, 0), level, level_dims).convert("RGB")

        # If the native level does not exactly match 1/16, resize
        if native_factor != SCALE_FACTOR:
            target_w = int(slide.dimensions[0] / SCALE_FACTOR)
            target_h = int(slide.dimensions[1] / SCALE_FACTOR)
            region = region.resize((target_w, target_h), Image.LANCZOS)

        # Output filename based on source filename
        out_name = Path(wsi_path).stem + "_1_16.png"
        out_path = Path(output_dir) / out_name
        region.save(out_path, "PNG")

        print(f"[OK] {wsi_path.name} -> {out_path.name} "
              f"({region.size[0]}x{region.size[1]}, level {level})")

    except Exception as e:
        print(f"[ERROR] Processing {wsi_path}: {e}")
    finally:
        slide.close()


def main():
    if len(sys.argv) != 3:
        print("Usage: python thumbnails.py <wsi_folder> <output_folder>")
        sys.exit(1)

    wsi_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common WSI extensions
    extensions = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".vms", ".scn"}

    wsi_files = [f for f in wsi_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in extensions]

    if not wsi_files:
        print(f"No WSI files found in {wsi_dir}")
        return

    print(f"{len(wsi_files)} WSI file(s) found. Processing...\n")
    for wsi in wsi_files:
        wsi_to_png(wsi, out_dir)


if __name__ == "__main__":
    main()