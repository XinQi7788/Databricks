
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.mil.abmil import ABMIL

class TileFeatureBags(Dataset):
    def __init__(self, tiles_dir: Path, feat_dir: Path, tme_csv: Path):
        self.tiles_dir = Path(tiles_dir)
        self.feat_dir = Path(feat_dir)
        self.labels = pd.read_csv(tme_csv)
        self.slide_to_label = dict(zip(self.labels['sample'], self.labels['tme_label']))
        self.E = np.load(self.feat_dir/'embeddings.npy')
        with open(self.feat_dir/'tiles.txt') as f:
            self.tiles = [l.strip() for l in f]
        self.slides = [t.split('__')[0] for t in self.tiles]
        self.slide_index = {}
        for i, s in enumerate(self.slides):
            self.slide_index.setdefault(s, []).append(i)
        self.slide_ids = [s for s in self.slide_index.keys() if s in self.slide_to_label]
        self.le = LabelEncoder().fit(self.labels['tme_label'])

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, i):
        s = self.slide_ids[i]
        idx = self.slide_index[s]
        x = torch.tensor(self.E[idx], dtype=torch.float32)
        y = torch.tensor(self.le.transform([self.slide_to_label[s]])[0], dtype=torch.long)
        return x, y, s


def train(model, loader, device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x = x[0].to(device)
        y = y.to(device)
        opt.zero_grad()
        logits, A = model(x)
        loss = ce(logits.unsqueeze(0), y)
        loss.backward(); opt.step()

@torch.no_grad()

def eval_model(model, loader, device):
    model.eval()
    correct = 0; total = 0
    for x, y, _ in loader:
        x = x[0].to(device)
        y = y.to(device)
        logits, A = model(x)
        pred = logits.argmax().item()
        correct += int(pred == y.item()); total += 1
    return correct/total if total>0 else 0.0


def main(tiles_dir, feat_dir, tme_csv, out_dir, max_epochs=10, hid_dim=256):
    ds = TileFeatureBags(Path(tiles_dir), Path(feat_dir), Path(tme_csv))
    train_ids, val_ids = train_test_split(ds.slide_ids, test_size=0.2, random_state=42)

    def subset(ids):
        idx_map = {s:i for i,s in enumerate(ds.slide_ids)}
        class SubsetDS(torch.utils.data.Dataset):
            def __init__(self, parent, ids):
                self.parent = parent; self.ids = ids
            def __len__(self): return len(self.ids)
            def __getitem__(self, i):
                return self.parent[idx_map[self.ids[i]]]
        return SubsetDS(ds, ids)

    tr_loader = DataLoader(subset(train_ids), batch_size=1, shuffle=True)
    va_loader = DataLoader(subset(val_ids), batch_size=1, shuffle=False)

    in_dim = ds.E.shape[1]
    n_classes = len(ds.le.classes_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ABMIL(in_dim, hid_dim=hid_dim, n_classes=n_classes).to(device)

    for epoch in range(max_epochs):
        train(model, tr_loader, device)
        acc = eval_model(model, va_loader, device)
        print(f"Epoch {epoch+1}/{max_epochs} val_acc={acc:.3f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'classes': ds.le.classes_.tolist()}, Path(out_dir)/'abmil_tme.pt')

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles_dir', required=True)
    ap.add_argument('--feat_dir', required=True)
    ap.add_argument('--tme_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_epochs', type=int, default=10)
    ap.add_argument('--hid_dim', type=int, default=256)
    args = ap.parse_args()
    main(args.tiles_dir, args.feat_dir, args.tme_csv, args.out_dir, args.max_epochs, args.hid_dim)
