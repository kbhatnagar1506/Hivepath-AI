"""
Risk Shaper Service
Edge-level risk/cost shaper that adjusts OSRM time matrix before solving
"""

import torch
import numpy as np
import os
from typing import List, Dict, Any

ART = "mlartifacts/risk_edge.pt"

class RiskShaper:
    """Edge-level risk shaper for routing optimization"""
    
    def __init__(self):
        self.model = None
        self.stops = None
        self.sid = None
        
        if os.path.exists(ART):
            try:
                ck = torch.load(ART, map_location="cpu", weights_only=False)
                self.stops = ck["stops"]
                self.sid = {s: i for i, s in enumerate(self.stops)}
                self.model = self._create_model()
                self.model.load_state_dict(ck["state_dict"])
                self.model.eval()
                print(f"✅ Loaded risk shaper model (MAE: {ck.get('val_mae', 'N/A'):.3f})")
            except Exception as e:
                print(f"⚠️  Failed to load risk shaper model: {e}")
                self.model = None
        else:
            print("⚠️  No risk shaper model found - using identity multipliers")
    
    def _create_model(self):
        """Create the risk prediction model architecture"""
        import torch.nn as nn
        
        class PairMLP(nn.Module):
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
                import torch.nn.functional as F
                hi = self.enc_i(X[i])
                hj = self.enc_j(X[j])
                z = torch.cat([hi, hj, tab], dim=1)
                return F.softplus(self.head(z)).squeeze(-1)
        
        return PairMLP()
    
    def shape(self, stops_order: List[str], osrm_min_matrix: List[List[float]], 
              hour: int, weekday: int, features_by_stop: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Shape the time matrix with risk multipliers
        
        Args:
            stops_order: List of stop IDs in same order as matrix (0=depot, 1..N=stops)
            osrm_min_matrix: Original OSRM time matrix (N+1 x N+1)
            hour: Current hour (0-23)
            weekday: Current weekday (0=Monday, 6=Sunday)
            features_by_stop: Dict mapping stop_id to {risk, light, cong} features
        
        Returns:
            Multiplier matrix M (N+1 x N+1), with zeros on diagonal
        """
        n = len(stops_order)
        M = np.zeros((n, n), dtype=np.float32)
        
        if self.model is None:
            return M
        
        # Build node feature matrix X (depot -> average stats)
        def get_features(sid):
            f = features_by_stop.get(sid, {"risk": 0.5, "light": 0.5, "cong": 0.5})
            return [f["risk"], f["light"], f["cong"]]
        
        X = torch.tensor([get_features(s) for s in stops_order], dtype=torch.float32)
        
        # Collect edge pairs for prediction
        pairs = []
        tabs = []
        is_idx, js_idx = [], []
        
        for i in range(n):
            for j in range(n):
                if i == j:  # Skip diagonal
                    continue
                
                base = osrm_min_matrix[i][j]
                if base <= 0 or base >= 1e6:  # Skip invalid edges
                    continue
                
                fi = features_by_stop.get(stops_order[i], {"risk": 0.5, "light": 0.5, "cong": 0.5})
                fj = features_by_stop.get(stops_order[j], {"risk": 0.5, "light": 0.5, "cong": 0.5})
                
                # Tabular features: [osrm_min, weekday, hour, risk_i, risk_j, light_i, light_j, cong_i, cong_j, incident_ij]
                tab = [
                    base, weekday, hour,
                    fi["risk"], fj["risk"],
                    fi["light"], fj["light"],
                    fi["cong"], fj["cong"],
                    0  # No incident data available in real-time
                ]
                
                tabs.append(tab)
                is_idx.append(i)
                js_idx.append(j)
        
        if not tabs:
            return M
        
        # Make predictions
        TAB = torch.tensor(np.array(tabs, dtype=np.float32))
        with torch.no_grad():
            pred = self.model(X, torch.tensor(is_idx), torch.tensor(js_idx), TAB).numpy()
        
        # Fill multiplier matrix
        for k, (i, j) in enumerate(zip(is_idx, js_idx)):
            M[i, j] = float(pred[k])
        
        return M
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "loaded": self.model is not None,
            "stops": self.stops,
            "num_stops": len(self.stops) if self.stops else 0
        }

# Global singleton instance
risk_shaper_singleton = RiskShaper()

def test_risk_shaper():
    """Test the risk shaper"""
    print("🧪 Testing Risk Shaper")
    print("=" * 30)
    
    # Test data
    stops_order = ["D", "S_A", "S_B", "S_C"]
    osrm_matrix = [
        [0, 10, 15, 20],
        [10, 0, 8, 12],
        [15, 8, 0, 6],
        [20, 12, 6, 0]
    ]
    
    features = {
        "D": {"risk": 0.4, "light": 0.7, "cong": 0.5},
        "S_A": {"risk": 0.6, "light": 0.8, "cong": 0.3},
        "S_B": {"risk": 0.3, "light": 0.4, "cong": 0.7},
        "S_C": {"risk": 0.8, "light": 0.6, "cong": 0.4}
    }
    
    hour = 14
    weekday = 2
    
    # Get model info
    info = risk_shaper_singleton.get_model_info()
    print(f"📊 Model Info:")
    print(f"   🎯 Loaded: {info['loaded']}")
    print(f"   📍 Stops: {info['num_stops']}")
    
    # Test shaping
    print(f"\n🔮 Testing risk shaping...")
    print(f"   📊 Original matrix shape: {np.array(osrm_matrix).shape}")
    
    M = risk_shaper_singleton.shape(stops_order, osrm_matrix, hour, weekday, features)
    
    print(f"   📊 Multiplier matrix shape: {M.shape}")
    print(f"   📊 Multiplier range: {M.min():.3f} - {M.max():.3f}")
    print(f"   📊 Non-zero multipliers: {np.count_nonzero(M)}")
    
    # Show some examples
    print(f"\n📋 Sample multipliers:")
    for i in range(len(stops_order)):
        for j in range(len(stops_order)):
            if i != j and M[i, j] > 0:
                print(f"   {stops_order[i]} → {stops_order[j]}: {M[i, j]:.3f}x")
    
    # Apply multipliers
    shaped_matrix = np.array(osrm_matrix) * (1.0 + M)
    
    print(f"\n📊 Shaped matrix:")
    print(f"   Original: {osrm_matrix[0][1]:.1f}min")
    print(f"   Shaped:   {shaped_matrix[0][1]:.1f}min")
    print(f"   Multiplier: {1.0 + M[0][1]:.3f}x")
    
    print(f"\n✅ Risk shaper test complete!")

if __name__ == "__main__":
    test_risk_shaper()
