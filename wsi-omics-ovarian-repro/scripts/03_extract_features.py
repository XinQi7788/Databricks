
import argparse
from pathlib import Path


def run_gigapath(tiles_dir, out_dir, batch_size, device):
    from src.features.gigapath_extractor import embed_folder
    embed_folder(Path(tiles_dir), Path(out_dir), batch_size=batch_size, device=device)


def run_chief(tiles_dir, out_dir, batch_size):
    from src.features.chief_extractor import main as chief_main
    chief_main(Path(tiles_dir), Path(out_dir), batch_size=batch_size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', choices=['gigapath','chief'], required=True)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    if args.model == 'gigapath':
        run_gigapath(args.tiles_dir, args.out_dir, args.batch_size, args.device)
    else:
        run_chief(args.tiles_dir, args.out_dir, args.batch_size)
