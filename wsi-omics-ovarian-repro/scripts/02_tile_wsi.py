
import argparse
from pathlib import Path
import openslide
from PIL import Image
import numpy as np

# Simple WSI tiler targeting a desired microns-per-pixel (mpp)

def get_mpp(slide: openslide.OpenSlide):
    try:
        mppx = float(slide.properties.get('openslide.mpp-x', '0'))
        mppy = float(slide.properties.get('openslide.mpp-y', '0'))
        if mppx > 0 and mppy > 0:
            return (mppx + mppy) / 2.0
    except Exception:
        return None
    return None


def choose_level_for_target_mpp(slide, target_mpp):
    base_mpp = get_mpp(slide)
    if base_mpp is None or target_mpp is None:
        return 0, 1.0
    best_level = 0
    best_diff = 1e9
    for lvl in range(slide.level_count):
        down = slide.level_downsamples[lvl]
        lvl_mpp = base_mpp * down
        diff = abs(lvl_mpp - target_mpp)
        if diff < best_diff:
            best_diff = diff
            best_level = lvl
    downsample = slide.level_downsamples[best_level]
    return best_level, downsample


def tile_wsi(wsi_path, out_dir, tile_size=256, step=256, target_mpp=0.5, fmt='png'):
    slide = openslide.OpenSlide(str(wsi_path))
    level, down = choose_level_for_target_mpp(slide, target_mpp)

    W, H = slide.level_dimensions[level]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basename = Path(wsi_path).stem
    index_lines = []

    for y in range(0, H - tile_size + 1, step):
        for x in range(0, W - tile_size + 1, step):
            region = slide.read_region((int(x*down), int(y*down)), level, (tile_size, tile_size)).convert('RGB')
            arr = np.array(region)
            if (arr.mean() > 230):  # skip near-white tiles
                continue
            tile_name = f"{basename}__{level}_{x}_{y}.{fmt}"
            region.save(out_dir / tile_name)
            index_lines.append(f"{tile_name},{basename},{level},{x},{y}
")

    with open(out_dir / f"{basename}__tiles_index.csv", 'w') as f:
        f.write("tile,slide,level,x,y
")
        f.writelines(index_lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tile_size', type=int, default=256)
    ap.add_argument('--step', type=int, default=256)
    ap.add_argument('--target_mpp', type=float, default=0.5)
    args = ap.parse_args()

    wsi_dir = Path(args.wsi_dir)
    for wsi in sorted(wsi_dir.glob('**/*')):
        if wsi.suffix.lower() in ['.svs', '.tiff', '.tif', '.ndpi', '.mrxs']:
            out_sub = Path(args.out_dir) / wsi.stem
            print(f"Tiling {wsi} -> {out_sub}")
            tile_wsi(wsi, out_sub, args.tile_size, args.step, args.target_mpp)
