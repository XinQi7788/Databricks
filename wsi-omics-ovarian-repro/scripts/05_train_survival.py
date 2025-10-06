# scripts/05_train_survival.py
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Dataset for survival
class SurvivalDataset(Dataset):
    def __init__(self, features, times, events):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return self.features[idx], self.times[idx], self.events[idx]

# Weibull MLP
class WeibullMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # outputs log(scale), log(shape)
        )

    def forward(self, x):
        out = self.net(x)
        log_scale, log_shape = out[:, 0], out[:, 1]
        scale = torch.exp(log_scale)
        shape = torch.exp(log_shape)
        return scale, shape

# Weibull NLL
def weibull_nll(scale, shape, time, event):
    eps = 1e-6
    loglik = event * (torch.log(shape + eps) + torch.log(scale + eps) +
                      (shape - 1) * torch.log(time + eps) -
                      (scale * (time + eps) ** shape)) \
           + (1 - event) * (-scale * (time + eps) ** shape)
    return -loglik.mean()

def train_model(model, train_loader, val_loader, epochs, lr, device, out_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_cindex = 0.0

    for epoch in range(epochs):
        model.train()
        for x, t, e in train_loader:
            x, t, e = x.to(device), t.to(device), e.to(device)
            scale, shape = model(x)
            loss = weibull_nll(scale, shape, t, e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds, times, events = [], [], []
        with torch.no_grad():
            for x, t, e in val_loader:
                x = x.to(device)
                scale, shape = model(x)
                pred_time = (1 / scale.cpu().numpy()) ** (1 / shape.cpu().numpy())
                preds.extend(pred_time)
                times.extend(t.numpy())
                events.extend(e.numpy())

        cindex = concordance_index(times, preds, events)
        print(f"Epoch {epoch+1}/{epochs} - Val C-index: {cindex:.4f}")

        if cindex > best_cindex:
            best_cindex = cindex
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Best C-index: {best_cindex:.4f}\n")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    features_df = pd.read_csv(args.features_csv)
    surv_df = pd.read_csv(args.surv_csv)
    merged = pd.merge(features_df, surv_df, on="slide")

    X = merged.drop(columns=["slide", "time", "event"]).values
    time = merged["time"].values
    event = merged["event"].values

    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X, time, event, test_size=0.2, random_state=42
    )

    train_ds = SurvivalDataset(X_train, t_train, e_train)
    val_ds = SurvivalDataset(X_val, t_val, e_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = WeibullMLP(input_dim=X.shape[1])
    train_model(model, train_loader, val_loader, args.max_epochs, lr=1e-3,
                device="cuda" if torch.cuda.is_available() else "cpu",
                out_dir=args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--surv_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)
