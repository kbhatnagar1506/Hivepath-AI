#!/usr/bin/env python3
"""
Train Service-Time GNN (GraphSAGE) for Knowledge Graph integration
"""

import os, json, math, warnings
import pandas as pd, numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")

# Try PyG; fall back to MLP if unavailable
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, global_mean_pool
    PYG=True
except Exception as e:
    print("PyG not available -> training MLP fallback. (Install torch-geometric for GNN.)")
    PYG=False

DATA_DIR = "data"
DEVICE = "cpu"

def load_kg_and_visits():
    nodes = pd.read_csv(f"{DATA_DIR}/kg_nodes.csv")
    edges = pd.read_csv(f"{DATA_DIR}/kg_edges.csv")
    visits = pd.read_csv(f"{DATA_DIR}/visits.csv")
    # build node index
    nid = {n:i for i,n in enumerate(nodes["id"])}
    # feature vector per node (Stop: demand, access; Depot zeros)
    X = []
    for _,r in nodes.iterrows():
        f = json.loads(r["features_json"]) if isinstance(r["features_json"], str) else {}
        X.append([float(f.get("demand",0)), float(f.get("access_score",0.5))])
    X = torch.tensor(X, dtype=torch.float32)

    # edge index (undirected)
    ei = []
    for _,r in edges.iterrows():
        if r["src"] in nid and r["dst"] in nid:
            ei.append([nid[r["src"]], nid[r["dst"]]])
            ei.append([nid[r["dst"]], nid[r["src"]]])
    edge_index = torch.tensor(ei, dtype=torch.long).T if len(ei) else torch.zeros((2,0), dtype=torch.long)

    # supervised rows only for Stop nodes present in visits
    # features: demand, access_score, hour, weekday
    stop_rows = []
    for _,v in visits.iterrows():
        sid = v["stop_id"]
        if sid not in nid: continue
        idx = nid[sid]
        stop_rows.append({
            "idx": idx,
            "x_tab": [v["demand"], v["access_score"], v["hour"], v["weekday"]],
            "y": float(v["service_min_actual"])
        })
    df = pd.DataFrame(stop_rows)
    return nodes, X, edge_index, df

class SAGEReg(nn.Module):
    def __init__(self, in_node=2, in_tab=4, hid=64):
        super().__init__()
        self.g1 = SAGEConv(in_node, hid)
        self.g2 = SAGEConv(hid, hid)
        self.mlp = nn.Sequential(nn.Linear(hid+in_tab, 64), nn.ReLU(),
                                 nn.Linear(64, 1))
    def forward(self, x, edge_index, idx_batch, x_tab):
        h = torch.relu(self.g1(x, edge_index))
        h = torch.relu(self.g2(h, edge_index))
        hb = h[idx_batch]                          # gather node reps for batch indices
        z = torch.cat([hb, x_tab], dim=1)
        return self.mlp(z).squeeze(-1)

class MLPReg(nn.Module):
    def __init__(self, in_tab=6):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_tab, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64,1))
    def forward(self, x_tab):
        return self.mlp(x_tab).squeeze(-1)

def main():
    nodes, X, edge_index, df = load_kg_and_visits()
    # train/val split
    msk = np.random.rand(len(df)) < 0.85
    tr, va = df[msk], df[~msk]
    y_mean = tr["y"].mean()

    if PYG:
        model = SAGEReg(in_node=X.shape[1], in_tab=4).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        for epoch in range(200):
            model.train(); opt.zero_grad()
            xb = torch.tensor(np.vstack(tr["x_tab"].values), dtype=torch.float32)
            ib = torch.tensor(tr["idx"].values, dtype=torch.long)
            y  = torch.tensor(tr["y"].values, dtype=torch.float32)
            pred = model(X, edge_index, ib, xb)
            loss = torch.nn.functional.l1_loss(pred, y)  # MAE
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            xb = torch.tensor(np.vstack(va["x_tab"].values), dtype=torch.float32)
            ib = torch.tensor(va["idx"].values, dtype=torch.long)
            y  = torch.tensor(va["y"].values, dtype=torch.float32)
            pred = model(X, edge_index, ib, xb)
            mae = (pred - y).abs().mean().item()
        os.makedirs("mlartifacts", exist_ok=True)
        torch.save({"state_dict":model.state_dict(),
                    "X":X, "edge_index":edge_index,
                    "y_mean":y_mean}, "mlartifacts/service_time_gnn.pt")
        print(f"GNN trained. Val MAE ~ {mae:.2f} min. Saved mlartifacts/service_time_gnn.pt")
    else:
        # Fallback: pure MLP on tabular (node features + tab features)
        # join X (node demand/access) with tab [demand,access,hour,weekday] -> 6 fea
        def stack(df):
            Xnode = X[torch.tensor(df["idx"].values)].numpy()
            Xtab  = np.vstack(df["x_tab"].values)
            return np.hstack([Xnode, Xtab])
        Xtr, Xva = stack(tr), stack(va)
        ytr, yva = tr["y"].values, va["y"].values
        model = MLPReg(in_tab=Xtr.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        for epoch in range(200):
            model.train(); opt.zero_grad()
            pred = model(torch.tensor(Xtr, dtype=torch.float32))
            y = torch.tensor(ytr, dtype=torch.float32)
            loss = torch.nn.functional.l1_loss(pred, y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mae = (model(torch.tensor(Xva, dtype=torch.float32)).squeeze()-torch.tensor(yva)).abs().mean().item()
        os.makedirs("mlartifacts", exist_ok=True)
        torch.save({"state_dict":model.state_dict(), "y_mean":y_mean}, "mlartifacts/service_time_mlp.pt")
        print(f"MLP trained. Val MAE ~ {mae:.2f} min. Saved mlartifacts/service_time_mlp.pt")

if __name__ == "__main__":
    main()
