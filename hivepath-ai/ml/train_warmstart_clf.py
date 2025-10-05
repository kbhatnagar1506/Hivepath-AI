#!/usr/bin/env python3
"""
Train Warm-start Clusterer (GNN-C) for Knowledge Graph integration
"""

import os, pandas as pd, numpy as np, torch, torch.nn as nn
os.makedirs("mlartifacts", exist_ok=True)

if not os.path.exists("data/assign_history.csv"):
    print("❌ data/assign_history.csv not found. Run scripts/synthesize_assign_history.py first.")
    exit(1)

DF = pd.read_csv("data/assign_history.csv")
# pick the most common K (vehicles per run)
K = DF.groupby("run_id")["vehicle_id"].nunique().mode().iloc[0]
print(f"Training warm-start clusterer for K={K} vehicles")

# build per-run label mapping  vehicle_id -> [0..K-1]
labs = {}
for rid, grp in DF.groupby("run_id"):
    vs = sorted(grp["vehicle_id"].unique())
    labs[rid] = {v:i for i,v in enumerate(vs)}

# Debug: check label distribution
print(f"Label distribution: {[len(vs) for vs in labs.values()][:5]}...")
print(f"Max label: {max(max(labs[rid].values()) for rid in labs)}")
print(f"Expected K: {K}")

rows = []
for _,r in DF.iterrows():
    rid = r["run_id"]; y = labs[rid].get(r["vehicle_id"], 0)
    rows.append([r["lat"], r["lng"], r["demand"], r["priority"], r["hour"], r["weekday"], y])
X = torch.tensor(np.array([r[:-1] for r in rows], dtype=np.float32))
Y = torch.tensor(np.array([r[-1] for r in rows], dtype=np.int64))

msk = torch.rand(len(Y)) < 0.85
Xtr, Xva = X[msk], X[~msk]; Ytr, Yva = Y[msk], Y[~msk]

class Clf(nn.Module):
    def __init__(self, in_dim=6, K=K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(),
                                 nn.Linear(128,64), nn.ReLU(),
                                 nn.Linear(64,K))
    def forward(self,x): return self.net(x)

model = Clf(in_dim=X.shape[1], K=K)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

print("🚀 Training Warm-start Clusterer (GNN-C)...")
for ep in range(200):
    model.train(); opt.zero_grad()
    logits = model(Xtr)
    loss = nn.CrossEntropyLoss()(logits, Ytr)
    loss.backward(); opt.step()
    if ep % 50 == 0:
        print(f"  Epoch {ep}: Loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    acc = (model(Xva).argmax(1) == Yva).float().mean().item()
torch.save({"state_dict":model.state_dict(), "K":K}, "mlartifacts/warmstart_clf.pt")
print(f"✅ Saved mlartifacts/warmstart_clf.pt (val acc ~ {acc:.2%}, K={K})")
