#!/usr/bin/env python3
"""
Train Edge-Level Risk/Cost Shaper (GNN-B)
Predicts time multipliers for edge pairs to avoid risky routes
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

os.makedirs("mlartifacts", exist_ok=True)

# Try PyG; fall back to MLP if not available
try:
    from torch_geometric.nn import SAGEConv
    PYG = True
    print("✅ PyTorch Geometric available - training with GNN")
except Exception as e:
    print(f"⚠️  PyG not available ({e}) -> training MLP fallback")
    PYG = False

def load_edge_data():
    """Load and preprocess edge observation data"""
    print("📊 Loading edge observation data...")
    
    DF = pd.read_csv("data/edges_obs.csv")
    print(f"   📈 Loaded {len(DF)} edge observations")
    
    # Get unique stops
    stops = sorted(set(DF["src_id"]).union(set(DF["dst_id"])))
    sid = {s: i for i, s in enumerate(stops)}
    
    print(f"   📍 Found {len(stops)} unique locations")
    
    # Map stop IDs to indices
    DF["i"] = DF["src_id"].map(sid)
    DF["j"] = DF["dst_id"].map(sid)
    
    # Calculate target variable: y = max(0, observed_min/osrm_min - 1.0)
    DF["y"] = np.clip(DF["observed_min"] / np.maximum(1e-6, DF["osrm_min"]) - 1.0, 0, 5.0)
    
    print(f"   🎯 Target range: {DF['y'].min():.3f} - {DF['y'].max():.3f}")
    print(f"   🎯 Target mean: {DF['y'].mean():.3f}")
    
    return DF, stops, sid

def create_node_features(DF, stops):
    """Create node features by aggregating edge statistics"""
    print("🔧 Creating node features...")
    
    # Aggregate simple stats from rows for each stop
    def agg(name):
        return DF.groupby("src_id")[name].mean().reindex(stops).fillna(0.5).values
    
    # Node features: [risk, light, congestion]
    X = np.stack([
        agg("risk_i"),
        agg("light_i"), 
        agg("cong_i")
    ], axis=1)
    
    X = torch.tensor(X, dtype=torch.float32)
    print(f"   🎯 Node features: {X.shape}")
    
    return X

def create_edge_features(DF):
    """Create edge/tabular features"""
    print("🔧 Creating edge features...")
    
    # Edge/tabular features: [osrm_min, weekday, hour, risk_i, risk_j, light_i, light_j, cong_i, cong_j, incident_ij]
    TAB = torch.tensor(np.stack([
        DF["osrm_min"].values,
        DF["weekday"].values,
        DF["hour"].values,
        DF["risk_i"].values,
        DF["risk_j"].values,
        DF["light_i"].values,
        DF["light_j"].values,
        DF["cong_i"].values,
        DF["cong_j"].values,
        DF["incident_ij"].values
    ], axis=1), dtype=torch.float32)
    
    I = torch.tensor(DF["i"].values, dtype=torch.long)
    J = torch.tensor(DF["j"].values, dtype=torch.long)
    Y = torch.tensor(DF["y"].values, dtype=torch.float32)
    
    print(f"   🎯 Edge features: {TAB.shape}")
    print(f"   🔗 Edge pairs: {len(I)}")
    
    return TAB, I, J, Y

class PairMLP(nn.Module):
    """MLP model for edge pair risk prediction"""
    
    def __init__(self, in_node=3, in_tab=10, hid=64):
        super().__init__()
        self.enc_i = nn.Sequential(
            nn.Linear(in_node, hid),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.enc_j = nn.Sequential(
            nn.Linear(in_node, hid),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hid + in_tab, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X, i, j, tab):
        # Encode source and destination node features
        hi = self.enc_i(X[i])
        hj = self.enc_j(X[j])
        
        # Combine with tabular features
        z = torch.cat([hi, hj, tab], dim=1)
        
        # Predict risk multiplier (>= 0)
        return torch.nn.functional.softplus(self.head(z)).squeeze(-1)

def train_model(X, TAB, I, J, Y):
    """Train the risk prediction model"""
    print("🚀 Training risk prediction model...")
    
    # Train/validation split
    msk = torch.rand(len(Y)) < 0.85
    def split(t):
        return t[msk], t[~msk]
    
    Xi, Xo = X, X  # Same node matrix everywhere
    Ii, Io = split(I)
    Ji, Jo = split(J)
    Ti, To = split(TAB)
    Yi, Yo = split(Y)
    
    print(f"   📊 Train samples: {len(Yi)}")
    print(f"   📊 Val samples: {len(Yo)}")
    
    # Create model
    model = PairMLP(in_node=X.shape[1], in_tab=TAB.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("   🔄 Starting training...")
    for ep in range(300):
        model.train()
        opt.zero_grad()
        
        pred = model(Xi, Ii, Ji, Ti)
        loss = torch.nn.functional.l1_loss(pred, Yi)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Validation
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xo, Io, Jo, To)
                val_loss = torch.nn.functional.l1_loss(val_pred, Yo).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(val_loss)
                
                if ep % 50 == 0:
                    print(f"      Epoch {ep:3d}: Train MAE={loss.item():.3f}, Val MAE={val_loss:.3f}")
                
                # Early stopping
                if patience_counter >= 30:
                    print(f"      Early stopping at epoch {ep}")
                    break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(Xo, Io, Jo, To)
        val_mae = torch.nn.functional.l1_loss(val_pred, Yo).item()
        val_mse = torch.nn.functional.mse_loss(val_pred, Yo).item()
        val_rmse = torch.sqrt(torch.tensor(val_mse)).item()
    
    print(f"   ✅ Training complete!")
    print(f"      📊 Validation MAE: {val_mae:.3f}")
    print(f"      📊 Validation RMSE: {val_rmse:.3f}")
    
    return model, val_mae

def main():
    """Main training function"""
    print("🧠 RISK EDGE SHAPER TRAINING")
    print("=" * 40)
    print("Training MLP to predict edge risk multipliers...")
    print()
    
    # Load data
    DF, stops, sid = load_edge_data()
    
    # Create features
    X = create_node_features(DF, stops)
    TAB, I, J, Y = create_edge_features(DF)
    
    # Train model
    model, val_mae = train_model(X, TAB, I, J, Y)
    
    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "stops": stops,
        "val_mae": val_mae,
        "model_type": "mlp"
    }, "mlartifacts/risk_edge.pt")
    
    print(f"\n💾 Model saved to mlartifacts/risk_edge.pt")
    print(f"   📊 Validation MAE: {val_mae:.3f}")
    print(f"   📍 Stops: {len(stops)}")
    print(f"   🎯 Model type: MLP")
    
    print("\n✅ Risk shaper training complete!")
    print("   Ready for inference in routing optimization")

if __name__ == "__main__":
    main()
