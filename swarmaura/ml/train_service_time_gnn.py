#!/usr/bin/env python3
"""
Service-Time GNN Training Script
Trains a GraphSAGE model to predict service times with MLP fallback
"""

import os
import json
import math
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")

# Try PyG; fall back to MLP if unavailable
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, global_mean_pool
    PYG = True
    print("✅ PyTorch Geometric available - training GNN")
except Exception as e:
    print(f"⚠️  PyG not available ({e}) -> training MLP fallback")
    print("   Install torch-geometric for GNN: pip install torch-geometric")
    PYG = False

DATA_DIR = "data"
DEVICE = "cpu"

def load_kg_and_visits():
    """Load Knowledge Graph and visit data"""
    print("📊 Loading Knowledge Graph and visit data...")
    
    nodes = pd.read_csv(f"{DATA_DIR}/kg_nodes.csv")
    edges = pd.read_csv(f"{DATA_DIR}/kg_edges.csv")
    visits = pd.read_csv(f"{DATA_DIR}/visits.csv")
    
    print(f"   📍 Nodes: {len(nodes)}")
    print(f"   🔗 Edges: {len(edges)}")
    print(f"   📈 Visits: {len(visits)}")
    
    # Build node index
    nid = {n: i for i, n in enumerate(nodes["id"])}
    
    # Feature vector per node (Stop: demand, access; Depot zeros)
    X = []
    for _, r in nodes.iterrows():
        f = json.loads(r["features_json"]) if isinstance(r["features_json"], str) else {}
        X.append([float(f.get("demand", 0)), float(f.get("access_score", 0.5))])
    X = torch.tensor(X, dtype=torch.float32)
    
    print(f"   🎯 Node features: {X.shape[1]} dimensions")
    
    # Edge index (undirected)
    ei = []
    for _, r in edges.iterrows():
        if r["src"] in nid and r["dst"] in nid:
            ei.append([nid[r["src"]], nid[r["dst"]]])
            ei.append([nid[r["dst"]], nid[r["src"]]])
    edge_index = torch.tensor(ei, dtype=torch.long).T if len(ei) else torch.zeros((2, 0), dtype=torch.long)
    
    print(f"   🔗 Edge index: {edge_index.shape[1]} edges")
    
    # Supervised rows only for Stop nodes present in visits
    # features: demand, access_score, hour, weekday
    stop_rows = []
    for _, v in visits.iterrows():
        sid = v["stop_id"]
        if sid not in nid: 
            continue
        idx = nid[sid]
        stop_rows.append({
            "idx": idx,
            "x_tab": [v["demand"], v["access_score"], v["hour"], v["weekday"]],
            "y": float(v["service_min_actual"])
        })
    
    df = pd.DataFrame(stop_rows)
    print(f"   📊 Training samples: {len(df)}")
    
    return nodes, X, edge_index, df

class SAGEReg(nn.Module):
    """GraphSAGE regression model for service time prediction"""
    
    def __init__(self, in_node=2, in_tab=4, hid=64):
        super().__init__()
        self.g1 = SAGEConv(in_node, hid)
        self.g2 = SAGEConv(hid, hid)
        self.mlp = nn.Sequential(
            nn.Linear(hid + in_tab, 64), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, edge_index, idx_batch, x_tab):
        # Graph convolutions
        h = torch.relu(self.g1(x, edge_index))
        h = torch.relu(self.g2(h, edge_index))
        
        # Gather node representations for batch indices
        hb = h[idx_batch]
        
        # Combine graph features with tabular features
        z = torch.cat([hb, x_tab], dim=1)
        
        # Final prediction
        return self.mlp(z).squeeze(-1)

class MLPReg(nn.Module):
    """MLP regression model (fallback when PyG unavailable)"""
    
    def __init__(self, in_tab=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_tab, 128), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_tab):
        return self.mlp(x_tab).squeeze(-1)

def train_gnn_model(nodes, X, edge_index, df):
    """Train GraphSAGE model"""
    print("🧠 Training GraphSAGE model...")
    
    # Train/validation split
    msk = np.random.rand(len(df)) < 0.85
    tr, va = df[msk], df[~msk]
    y_mean = tr["y"].mean()
    
    print(f"   📊 Train samples: {len(tr)}")
    print(f"   📊 Val samples: {len(va)}")
    print(f"   📈 Mean service time: {y_mean:.2f} minutes")
    
    model = SAGEReg(in_node=X.shape[1], in_tab=4).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("   🚀 Starting training...")
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        
        # Training batch
        xb = torch.tensor(np.vstack(tr["x_tab"].values), dtype=torch.float32)
        ib = torch.tensor(tr["idx"].values, dtype=torch.long)
        y = torch.tensor(tr["y"].values, dtype=torch.float32)
        
        pred = model(X, edge_index, ib, xb)
        loss = torch.nn.functional.l1_loss(pred, y)  # MAE
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                xb_val = torch.tensor(np.vstack(va["x_tab"].values), dtype=torch.float32)
                ib_val = torch.tensor(va["idx"].values, dtype=torch.long)
                y_val = torch.tensor(va["y"].values, dtype=torch.float32)
                pred_val = model(X, edge_index, ib_val, xb_val)
                val_loss = (pred_val - y_val).abs().mean().item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(val_loss)
                
                if epoch % 50 == 0:
                    print(f"      Epoch {epoch:3d}: Train MAE={loss.item():.3f}, Val MAE={val_loss:.3f}")
                
                # Early stopping
                if patience_counter >= 30:
                    print(f"      Early stopping at epoch {epoch}")
                    break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(np.vstack(va["x_tab"].values), dtype=torch.float32)
        ib = torch.tensor(va["idx"].values, dtype=torch.long)
        y = torch.tensor(va["y"].values, dtype=torch.float32)
        pred = model(X, edge_index, ib, xb)
        mae = (pred - y).abs().mean().item()
        mse = ((pred - y) ** 2).mean().item()
        rmse = math.sqrt(mse)
    
    print(f"   ✅ GNN training complete!")
    print(f"      📊 Validation MAE: {mae:.2f} minutes")
    print(f"      📊 Validation RMSE: {rmse:.2f} minutes")
    
    # Save model
    os.makedirs("mlartifacts", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "X": X, 
        "edge_index": edge_index,
        "y_mean": y_mean,
        "model_type": "gnn"
    }, "mlartifacts/service_time_gnn.pt")
    
    print(f"   💾 Model saved to mlartifacts/service_time_gnn.pt")
    
    return model, mae

def train_mlp_model(nodes, X, edge_index, df):
    """Train MLP model (fallback)"""
    print("🧠 Training MLP model (fallback)...")
    
    # Train/validation split
    msk = np.random.rand(len(df)) < 0.85
    tr, va = df[msk], df[~msk]
    y_mean = tr["y"].mean()
    
    print(f"   📊 Train samples: {len(tr)}")
    print(f"   📊 Val samples: {len(va)}")
    print(f"   📈 Mean service time: {y_mean:.2f} minutes")
    
    # Prepare features: node features + tabular features
    def stack_features(df_subset):
        Xnode = X[torch.tensor(df_subset["idx"].values)].numpy()
        Xtab = np.vstack(df_subset["x_tab"].values)
        return np.hstack([Xnode, Xtab])
    
    Xtr, Xva = stack_features(tr), stack_features(va)
    ytr, yva = tr["y"].values, va["y"].values
    
    print(f"   🎯 Feature dimensions: {Xtr.shape[1]}")
    
    model = MLPReg(in_tab=Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("   🚀 Starting training...")
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        
        pred = model(torch.tensor(Xtr, dtype=torch.float32))
        y = torch.tensor(ytr, dtype=torch.float32)
        loss = torch.nn.functional.l1_loss(pred, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(torch.tensor(Xva, dtype=torch.float32))
                val_loss = (pred_val - torch.tensor(yva)).abs().mean().item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(val_loss)
                
                if epoch % 50 == 0:
                    print(f"      Epoch {epoch:3d}: Train MAE={loss.item():.3f}, Val MAE={val_loss:.3f}")
                
                # Early stopping
                if patience_counter >= 30:
                    print(f"      Early stopping at epoch {epoch}")
                    break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xva, dtype=torch.float32))
        mae = (pred - torch.tensor(yva)).abs().mean().item()
        mse = ((pred - torch.tensor(yva)) ** 2).mean().item()
        rmse = math.sqrt(mse)
    
    print(f"   ✅ MLP training complete!")
    print(f"      📊 Validation MAE: {mae:.2f} minutes")
    print(f"      📊 Validation RMSE: {rmse:.2f} minutes")
    
    # Save model
    os.makedirs("mlartifacts", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "y_mean": y_mean,
        "model_type": "mlp"
    }, "mlartifacts/service_time_mlp.pt")
    
    print(f"   💾 Model saved to mlartifacts/service_time_mlp.pt")
    
    return model, mae

def main():
    """Main training function"""
    print("🧠 SERVICE-TIME GNN TRAINING")
    print("=" * 40)
    
    # Load data
    nodes, X, edge_index, df = load_kg_and_visits()
    
    if PYG:
        # Train GraphSAGE model
        model, mae = train_gnn_model(nodes, X, edge_index, df)
        print(f"\n🎉 GNN training complete! Validation MAE: {mae:.2f} minutes")
    else:
        # Train MLP model
        model, mae = train_mlp_model(nodes, X, edge_index, df)
        print(f"\n🎉 MLP training complete! Validation MAE: {mae:.2f} minutes")
    
    print("\n📊 Model Performance Summary:")
    print(f"   🎯 Model Type: {'GraphSAGE' if PYG else 'MLP'}")
    print(f"   📈 Validation MAE: {mae:.2f} minutes")
    print(f"   📊 Training Samples: {len(df)}")
    print(f"   🔗 Graph Edges: {edge_index.shape[1] if PYG else 'N/A'}")
    
    print("\n✅ Ready for inference!")
    print("   Use the service_time_model.py for predictions")

if __name__ == "__main__":
    main()
