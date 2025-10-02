
import argparse
from pathlib import Path
import timm
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Loads the Prov-GigaPath tile encoder from HF via timm

def load_gigapath_tile_encoder(device='cuda'):
    model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
    model.eval()
    model.to(device)
    return model

TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

@torch.no_grad()
def embed_folder(tiles_dir: Path, out_dir: Path, batch_size=256, device='cuda'):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_gigapath_tile_encoder(device)

    tiles = sorted([p for p in tiles_dir.glob('**/*.png')])
    feats = []
    names = []

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    for batch in tqdm(chunks(tiles, batch_size), total=(len(tiles)+batch_size-1)//batch_size):
        imgs = []
        for p in batch:
            img = Image.open(p).convert('RGB')
            imgs.append(TRANSFORM(img))
            names.append(p.name)
        x = torch.stack(imgs).to(device)
        z = model.forward_features(x)
        if hasattr(model, 'global_pool') and model.global_pool is not None:
            z = model.global_pool(z)
        z = z.detach().cpu().numpy()
        feats.append(z)

    feats = np.vstack(feats)
    np.save(out_dir / 'embeddings.npy', feats)
    with open(out_dir / 'tiles.txt', 'w') as f:
        for n in names:
            f.write(f"{n}
")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    embed_folder(Path(args.tiles_dir), Path(args.out_dir), args.batch_size, args.device)
