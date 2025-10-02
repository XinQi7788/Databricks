
import torch
import torch.nn as nn

class ABMIL(nn.Module):
    def __init__(self, in_dim, hid_dim=256, n_classes=4, gated=True, dropout=0.25):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        if gated:
            self.att_a = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Tanh())
            self.att_b = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())
            self.att = nn.Linear(hid_dim, 1)
        else:
            self.att = nn.Sequential(nn.Linear(hid_dim, 1))
        self.gated = gated
        self.cls = nn.Linear(hid_dim, n_classes)

    def attention(self, H):
        if self.gated:
            A = self.att_a(H) * self.att_b(H)
            A = self.att(A)
        else:
            A = self.att(H)
        A = torch.softmax(A, dim=0)  # [N,1]
        M = torch.sum(A * H, dim=0)  # [hid]
        return M, A

    def forward(self, X):
        H = self.feature(X)
        M, A = self.attention(H)
        logits = self.cls(M)
        return logits, A
