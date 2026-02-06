#!/usr/bin/env python3
"""
Train Warm-Start Clusterer (GNN-C)
Predicts cluster assignments for stops to seed OR-Tools with good initial routes
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

os.makedirs("mlartifacts", exist_ok=True)

def load_assignment_data():
    """Load and preprocess assignment history data"""
    print("📊 Loading assignment history data...")
    
    DF = pd.read_csv("data/assign_history.csv")
    print(f"   📈 Loaded {len(DF)} assignment records")
    
    # Pick the maximum K (vehicles per run) to handle all cases
    vehicle_counts = DF.groupby("run_id")["vehicle_id"].nunique()
    K = vehicle_counts.max()
    
    print(f"   🚛 Maximum vehicle count: {K}")
    print(f"   📊 Vehicle count distribution: {vehicle_counts.value_counts().to_dict()}")
    
    # Build per-run label mapping: vehicle_id -> [0..K-1]
    labs = {}
    for rid, grp in DF.groupby("run_id"):
        vs = sorted(grp["vehicle_id"].unique())
        labs[rid] = {v: i for i, v in enumerate(vs)}
    
    print(f"   📊 Runs: {len(labs)}")
    
    return DF, K, labs

def create_training_data(DF, K, labs):
    """Create training data for clustering"""
    print("🔧 Creating training data...")
    
    rows = []
    for _, r in DF.iterrows():
        rid = r["run_id"]
        y = labs[rid].get(r["vehicle_id"], 0)
        
        rows.append([
            r["lat"],
            r["lng"], 
            r["demand"],
            r["priority"],
            r["hour"],
            r["weekday"],
            y
        ])
    
    X = torch.tensor(np.array([r[:-1] for r in rows], dtype=np.float32))
    Y = torch.tensor(np.array([r[-1] for r in rows], dtype=np.int64))
    
    print(f"   🎯 Features: {X.shape}")
    print(f"   🎯 Labels: {Y.shape}")
    print(f"   🎯 Classes: {K}")
    
    # Show class distribution
    unique, counts = torch.unique(Y, return_counts=True)
    print(f"   📊 Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    return X, Y, K

class ClusteringClassifier(nn.Module):
    """MLP classifier for stop clustering"""
    
    def __init__(self, in_dim=6, K=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, K)
        )
    
    def forward(self, x):
        return self.net(x)

def train_model(X, Y, K):
    """Train the clustering classifier"""
    print("🚀 Training clustering classifier...")
    
    # Train/validation split
    msk = torch.rand(len(Y)) < 0.85
    Xtr, Xva = X[msk], X[~msk]
    Ytr, Yva = Y[msk], Y[~msk]
    
    print(f"   📊 Train samples: {len(Xtr)}")
    print(f"   📊 Val samples: {len(Xva)}")
    
    # Create model
    model = ClusteringClassifier(in_dim=X.shape[1], K=K)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("   🔄 Starting training...")
    for ep in range(200):
        model.train()
        opt.zero_grad()
        
        logits = model(Xtr)
        loss = nn.CrossEntropyLoss()(logits, Ytr)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Validation
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xva)
                val_pred = val_logits.argmax(1)
                val_acc = (val_pred == Yva).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(1 - val_acc)  # Minimize (1 - accuracy)
                
                if ep % 50 == 0:
                    print(f"      Epoch {ep:3d}: Train Loss={loss.item():.3f}, Val Acc={val_acc:.3f}")
                
                # Early stopping
                if patience_counter >= 30:
                    print(f"      Early stopping at epoch {ep}")
                    break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(Xva)
        val_pred = val_logits.argmax(1)
        val_acc = (val_pred == Yva).float().mean().item()
        
        # Per-class accuracy
        class_acc = []
        for k in range(K):
            mask = (Yva == k)
            if mask.sum() > 0:
                acc = (val_pred[mask] == Yva[mask]).float().mean().item()
                class_acc.append(acc)
            else:
                class_acc.append(0.0)
    
    print(f"   ✅ Training complete!")
    print(f"      📊 Validation Accuracy: {val_acc:.3f}")
    print(f"      📊 Per-class Accuracy: {[f'{acc:.3f}' for acc in class_acc]}")
    
    return model, val_acc

def main():
    """Main training function"""
    print("🧠 WARM-START CLUSTERER TRAINING")
    print("=" * 40)
    print("Training MLP to predict stop cluster assignments...")
    print()
    
    # Load data
    DF, K, labs = load_assignment_data()
    
    # Create training data
    X, Y, K = create_training_data(DF, K, labs)
    
    # Train model
    model, val_acc = train_model(X, Y, K)
    
    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "K": K,
        "val_acc": val_acc,
        "model_type": "mlp"
    }, "mlartifacts/warmstart_clf.pt")
    
    print(f"\n💾 Model saved to mlartifacts/warmstart_clf.pt")
    print(f"   📊 Validation Accuracy: {val_acc:.3f}")
    print(f"   🎯 Classes: {K}")
    print(f"   🎯 Model type: MLP")
    
    print("\n✅ Warm-start clusterer training complete!")
    print("   Ready for inference in routing optimization")

if __name__ == "__main__":
    main()
