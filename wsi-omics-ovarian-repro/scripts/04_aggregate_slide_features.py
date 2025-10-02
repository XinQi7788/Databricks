
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim//2), nn.Tanh(),
            nn.Linear(in_dim//2, 1)
        )
    def forward(self, X):
        a = self.attn(X)  # [N,1]
        w = torch.softmax(a, dim=0)
        z = (w * X).sum(dim=0)
        return z, w


def load_embeddings(feat_dir: Path):
    E = np.load(feat_dir / 'embeddings.npy')
    with open(feat_dir / 'tiles.txt') as f:
        tiles = [l.strip() for l in f]
    slides = [t.split('__')[0] for t in tiles]
    return E, tiles, slides


def main(feat_dir, out_csv, pool='mean', epochs=1):
    feat_dir = Path(feat_dir)
    E, tiles, slides = load_embeddings(feat_dir)
    df = pd.DataFrame({'tile': tiles, 'slide': slides})

    out_rows = []
    D = E.shape[1]

    if pool == 'mean':
        for sid, idx in tqdm(df.groupby('slide').indices.items()):
            z = E[idx].mean(axis=0)
            out_rows.append([sid] + z.tolist())
    else:
        attn = AttentionPool(D)
        opt = torch.optim.Adam(attn.parameters(), lr=1e-4)
        X = torch.tensor(E, dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            z, w = attn(X)
            loss = ((X - z.unsqueeze(0))**2).mean()
            loss.backward(); opt.step()
        for sid, idx in tqdm(df.groupby('slide').indices.items()):
            x = torch.tensor(E[idx], dtype=torch.float32)
            with torch.no_grad():
                z, _ = attn(x)
            out_rows.append([sid] + z.numpy().tolist())

    cols = ['slide'] + [f'f{i}' for i in range(D)]
    out_df = pd.DataFrame(out_rows, columns=cols)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat_dir', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--pool', choices=['mean','attention'], default='mean')
    ap.add_argument('--epochs', type=int, default=1)
    args = ap.parse_args()
    main(args.feat_dir, args.out_csv, args.pool, args.epochs)
