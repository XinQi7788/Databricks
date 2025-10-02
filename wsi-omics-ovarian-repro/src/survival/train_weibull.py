
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Weibull NLL with right censoring

def weibull_nll(time, event, lam, k):
    eps = 1e-8
    t = time + eps
    l = lam + eps
    ke = k + eps
    log_h = torch.log(ke/l) + (ke-1.0)*torch.log(t/l)
    log_S = - (t/l)**ke
    return - (event*log_h + log_S)

class MLPWeibull(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
        )
        self.head = nn.Linear(hidden//2, 2)
    def forward(self, x):
        z = self.net(x)
        out = self.head(z)
        log_lam, log_k = out[:,0], out[:,1]
        lam = torch.exp(log_lam)
        k = torch.exp(log_k)
        return lam, k


def load_features(features_csv, surv_csv):
    X = pd.read_csv(features_csv)
    S = pd.read_csv(surv_csv)
    df = X.merge(S, on='slide')
    feats = df.drop(columns=['slide','time','event']).values.astype('float32')
    time = df['time'].values.astype('float32')
    event = df['event'].values.astype('float32')
    return feats, time, event


def main(features_csv, surv_csv, out_dir, max_epochs=50, lr=1e-3):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, T, E = load_features(features_csv, surv_csv)

    Xtr, Xte, Ttr, Tte, Etr, Ete = train_test_split(X, T, E, test_size=0.2, random_state=42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPWeibull(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(Xtr).to(device)
    Ttr = torch.tensor(Ttr).to(device)
    Etr = torch.tensor(Etr).to(device)

    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        lam, k = model(Xtr)
        loss = weibull_nll(Ttr, Etr, lam, k).mean()
        loss.backward(); opt.step()
        if (epoch+1)%10==0:
            print(f"epoch {epoch+1}/{max_epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte).to(device)
        lam, k = model(Xte_t)
        med = lam*(np.log(2))**(1.0/torch.clamp(k,min=1e-3))
        risk = (-med).detach().cpu().numpy().ravel()
    c = concordance_index(Tte, -risk, Ete)
    with open(out_dir/'metrics.txt','w') as f:
        f.write(f"c_index={c:.4f}
")
    torch.save(model.state_dict(), out_dir/'weibull_model.pt')
    print(f"Saved to {out_dir}. c-index={c:.4f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_csv', required=True)
    ap.add_argument('--surv_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    main(args.features_csv, args.surv_csv, args.out_dir, args.max_epochs, args.lr)
