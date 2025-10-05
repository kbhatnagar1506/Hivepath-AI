#!/usr/bin/env python3
"""
Train Edge-level Risk/Cost Shaper (GNN-B) for Knowledge Graph integration
"""

import os, pandas as pd, numpy as np, torch, torch.nn as nn
os.makedirs("mlartifacts", exist_ok=True)

# Try PyG; fall back to MLP if not available
try:
    from torch_geometric.nn import SAGEConv
    PYG=True
except Exception:
    PYG=False

# ---- load data -------------------------------------------------------------
if not os.path.exists("data/edges_obs.csv"):
    print("❌ data/edges_obs.csv not found. Run scripts/build_boston_pack.py first.")
    exit(1)

DF = pd.read_csv("data/edges_obs.csv")
stops = sorted(set(DF["src_id"]).union(set(DF["dst_id"])))
sid = {s:i for i,s in enumerate(stops)}
DF["i"] = DF["src_id"].map(sid)
DF["j"] = DF["dst_id"].map(sid)
DF["y"] = np.clip(DF["observed_min"]/np.maximum(1e-6, DF["osrm_min"]) - 1.0, 0, 5.0)

# Node features per stop (aggregate simple stats from rows)
def agg(name): return DF.groupby("src_id")[name].mean().reindex(stops).fillna(0.5).values
X = np.stack([
    agg("risk_i"), agg("light_i"), agg("cong_i")
], axis=1)
X = torch.tensor(X, dtype=torch.float32)

# Edge/tab features
TAB = torch.tensor(np.stack([
    DF["osrm_min"].values,
    DF["weekday"].values, DF["hour"].values,
    DF["risk_i"].values, DF["risk_j"].values,
    DF["light_i"].values, DF["light_j"].values,
    DF["cong_i"].values, DF["cong_j"].values,
    DF["incident_ij"].values
], axis=1), dtype=torch.float32)
I = torch.tensor(DF["i"].values, dtype=torch.long)
J = torch.tensor(DF["j"].values, dtype=torch.long)
Y = torch.tensor(DF["y"].values, dtype=torch.float32)

msk = torch.rand(len(Y)) < 0.85
def split(t): return t[msk], t[~msk]
Xi, Xo = X, X  # same node matrix everywhere
Ii, Io = split(I); Ji, Jo = split(J); Ti, To = split(TAB); Yi, Yo = split(Y)

# ---- models ----------------------------------------------------------------
class PairMLP(nn.Module):
    def __init__(self, in_node=3, in_tab=10, hid=64):
        super().__init__()
        self.enc_i = nn.Sequential(nn.Linear(in_node, hid), nn.ReLU())
        self.enc_j = nn.Sequential(nn.Linear(in_node, hid), nn.ReLU())
        self.head  = nn.Sequential(nn.Linear(2*hid+in_tab, 64), nn.ReLU(),
                                   nn.Linear(64, 1))
    def forward(self, X, i, j, tab):
        hi = self.enc_i(X[i]); hj = self.enc_j(X[j])
        z  = torch.cat([hi, hj, tab], dim=1)
        return torch.nn.functional.softplus(self.head(z)).squeeze(-1)  # >=0

if PYG:
    # Optionally learn node embeddings with a tiny GNN; here we keep simple MLP for speed
    pass

model = PairMLP(in_node=X.shape[1], in_tab=TAB.shape[1])
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

print("🚀 Training Edge Risk Shaper (GNN-B)...")
for ep in range(300):
    model.train(); opt.zero_grad()
    pred = model(Xi, Ii, Ji, Ti)
    loss = torch.nn.functional.l1_loss(pred, Yi)
    loss.backward(); opt.step()
    if ep % 50 == 0:
        print(f"  Epoch {ep}: Loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    val_mae = torch.nn.functional.l1_loss(model(Xo, Io, Jo, To), Yo).item()
torch.save({"state_dict":model.state_dict(),
           "stops":stops, "val_mae":val_mae},
           "mlartifacts/risk_edge.pt")
print(f"✅ Saved mlartifacts/risk_edge.pt (val MAE ~ {val_mae:.3f})")
