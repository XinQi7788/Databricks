
import argparse
import os
import subprocess
from pathlib import Path

# Thin wrapper that calls CHIEF's patch feature extractor script.
# Set CHIEF_REPO to your local clone of hms-dbmi/CHIEF.

def main(tiles_dir: Path, out_dir: Path, batch_size: int = 256):
    chief_repo = os.environ.get('CHIEF_REPO')
    if not chief_repo:
        raise RuntimeError('Please set CHIEF_REPO to your local path of hms-dbmi/CHIEF')
    script = Path(chief_repo) / 'Get_CHIEF_patch_feature.py'
    if not script.exists():
        raise FileNotFoundError(f'Cannot find {script}. Did you clone the CHIEF repo?')

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        'python', str(script),
        '--img_path', str(tiles_dir),
        '--save_path', str(out_dir),
        '--batch_size', str(batch_size)
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=256)
    args = ap.parse_args()
    main(Path(args.tiles_dir), Path(args.out_dir), args.batch_size)
