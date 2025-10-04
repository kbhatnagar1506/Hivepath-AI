"""
Service Time Prediction Model
Inference service for GNN/MLP service time prediction
"""

import torch
import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Model artifact paths
ART_GNN = "mlartifacts/service_time_gnn.pt"
ART_MLP = "mlartifacts/service_time_mlp.pt"

class ServiceTimePredictor:
    """Service time prediction using trained GNN or MLP models"""
    
    def __init__(self):
        self.mode = None
        self.model = None
        self.X = None
        self.edge_index = None
        self.y_mean = None
        self.node_id_to_idx = None
        
        # Try to load GNN model first
        if os.path.exists(ART_GNN):
            try:
                self._load_gnn_model()
                self.mode = "gnn"
                print("✅ Loaded GraphSAGE model for service time prediction")
            except Exception as e:
                print(f"⚠️  Failed to load GNN model: {e}")
                self._try_load_mlp()
        else:
            self._try_load_mlp()
    
    def _load_gnn_model(self):
        """Load trained GraphSAGE model"""
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError:
            raise ImportError("PyTorch Geometric not available")
        
        # Load checkpoint
        ckpt = torch.load(ART_GNN, map_location="cpu", weights_only=False)
        self.X = ckpt["X"]
        self.edge_index = ckpt["edge_index"]
        self.y_mean = ckpt["y_mean"]
        
        # Recreate model architecture
        class SAGEReg(torch.nn.Module):
            def __init__(self, in_node=2, in_tab=4, hid=64):
                super().__init__()
                self.g1 = SAGEConv(in_node, hid)
                self.g2 = SAGEConv(hid, hid)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(hid + in_tab, 64), 
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(32, 1)
                )
                
            def forward(self, x, ei, idxb, xt):
                h = torch.relu(self.g1(x, ei))
                h = torch.relu(self.g2(h, ei))
                hb = h[idxb]
                return self.mlp(torch.cat([hb, xt], 1)).squeeze(-1)
        
        self.model = SAGEReg().eval()
        self.model.load_state_dict(ckpt["state_dict"])
        
        # Build node ID to index mapping
        self._build_node_mapping()
    
    def _try_load_mlp(self):
        """Try to load MLP model as fallback"""
        if os.path.exists(ART_MLP):
            try:
                self._load_mlp_model()
                self.mode = "mlp"
                print("✅ Loaded MLP model for service time prediction")
            except Exception as e:
                print(f"⚠️  Failed to load MLP model: {e}")
                self.mode = "none"
                print("⚠️  No trained model available - using heuristic fallback")
        else:
            self.mode = "none"
            print("⚠️  No trained model available - using heuristic fallback")
    
    def _load_mlp_model(self):
        """Load trained MLP model"""
        ckpt = torch.load(ART_MLP, map_location="cpu", weights_only=False)
        self.y_mean = ckpt["y_mean"]
        
        # Recreate model architecture
        class MLPReg(torch.nn.Module):
            def __init__(self, in_dim=6):
                super().__init__()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, 128), 
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(128, 64), 
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 1)
                )
                
            def forward(self, x):
                return self.mlp(x).squeeze(-1)
        
        self.model = MLPReg().eval()
        self.model.load_state_dict(ckpt["state_dict"])
    
    def _build_node_mapping(self):
        """Build mapping from node IDs to indices"""
        if self.mode == "gnn" and self.X is not None:
            # Load the original node data to build mapping
            try:
                nodes_df = pd.read_csv("data/kg_nodes.csv")
                self.node_id_to_idx = {row["id"]: idx for idx, row in nodes_df.iterrows()}
            except FileNotFoundError:
                print("⚠️  Could not load node mapping - using default indices")
                self.node_id_to_idx = {}
    
    def predict_minutes(self, stop_rows: List[Dict[str, Any]]) -> List[float]:
        """
        Predict service times for a list of stops
        
        Args:
            stop_rows: List of dicts with keys:
                - id: stop ID
                - demand: demand value
                - access_score: accessibility score (0-1)
                - hour: hour of day (0-23)
                - weekday: day of week (0=Monday, 6=Sunday)
                - node_idx: optional node index for GNN
        
        Returns:
            List of predicted service times in minutes
        """
        if self.mode == "none":
            return self._heuristic_predictions(stop_rows)
        elif self.mode == "gnn":
            return self._gnn_predictions(stop_rows)
        else:  # mlp
            return self._mlp_predictions(stop_rows)
    
    def _heuristic_predictions(self, stop_rows: List[Dict[str, Any]]) -> List[float]:
        """Heuristic fallback predictions"""
        predictions = []
        for r in stop_rows:
            demand = r.get("demand", 120)
            access_score = r.get("access_score", 0.6)
            
            # Simple heuristic based on demand and access
            base_time = 4.0 + 0.06 * demand
            access_penalty = 5.0 * (1.0 - access_score)
            service_time = max(3.0, base_time + access_penalty)
            
            predictions.append(round(service_time, 1))
        
        return predictions
    
    def _gnn_predictions(self, stop_rows: List[Dict[str, Any]]) -> List[float]:
        """GraphSAGE model predictions"""
        if self.model is None or self.X is None or self.edge_index is None:
            return self._heuristic_predictions(stop_rows)
        
        # Prepare batch data
        idx_list = []
        x_tab_list = []
        
        for r in stop_rows:
            # Get node index
            stop_id = r.get("id", "")
            if self.node_id_to_idx and stop_id in self.node_id_to_idx:
                node_idx = self.node_id_to_idx[stop_id]
            else:
                node_idx = r.get("node_idx", 0)
            
            idx_list.append(node_idx)
            x_tab_list.append([
                r.get("demand", 120),
                r.get("access_score", 0.6),
                r.get("hour", 10),
                r.get("weekday", 2)
            ])
        
        # Convert to tensors
        idx_tensor = torch.tensor(idx_list, dtype=torch.long)
        x_tab_tensor = torch.tensor(x_tab_list, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(self.X, self.edge_index, idx_tensor, x_tab_tensor)
            predictions = predictions.clamp_min(3.0)  # Minimum 3 minutes
        
        return [round(p.item(), 1) for p in predictions]
    
    def _mlp_predictions(self, stop_rows: List[Dict[str, Any]]) -> List[float]:
        """MLP model predictions"""
        if self.model is None:
            return self._heuristic_predictions(stop_rows)
        
        # Prepare features: node features + tabular features
        # For MLP, we duplicate the tabular features as node features
        x_tab_list = []
        
        for r in stop_rows:
            demand = r.get("demand", 120)
            access_score = r.get("access_score", 0.6)
            hour = r.get("hour", 10)
            weekday = r.get("weekday", 2)
            
            # MLP input: [node_demand, node_access, tab_demand, tab_access, hour, weekday]
            x_tab_list.append([demand, access_score, demand, access_score, hour, weekday])
        
        x_tab_tensor = torch.tensor(x_tab_list, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(x_tab_tensor)
            predictions = predictions.clamp_min(3.0)  # Minimum 3 minutes
        
        return [round(p.item(), 1) for p in predictions]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "mode": self.mode,
            "model_type": "GraphSAGE" if self.mode == "gnn" else "MLP" if self.mode == "mlp" else "Heuristic",
            "y_mean": self.y_mean,
            "has_graph": self.edge_index is not None and self.edge_index.shape[1] > 0,
            "num_nodes": self.X.shape[0] if self.X is not None else 0,
            "num_edges": self.edge_index.shape[1] if self.edge_index is not None else 0
        }

# Global singleton instance
predictor_singleton = ServiceTimePredictor()

def test_service_time_predictor():
    """Test the service time predictor"""
    print("🧪 Testing Service Time Predictor")
    print("=" * 40)
    
    # Test data
    test_stops = [
        {
            "id": "S_A",
            "demand": 150,
            "access_score": 0.72,
            "hour": 10,
            "weekday": 2
        },
        {
            "id": "S_B", 
            "demand": 140,
            "access_score": 0.61,
            "hour": 14,
            "weekday": 2
        },
        {
            "id": "S_C",
            "demand": 160,
            "access_score": 0.55,
            "hour": 16,
            "weekday": 2
        }
    ]
    
    # Get model info
    info = predictor_singleton.get_model_info()
    print(f"📊 Model Info:")
    print(f"   🎯 Type: {info['model_type']}")
    print(f"   📈 Mean Service Time: {info['y_mean']:.2f} minutes" if info['y_mean'] else "   📈 Mean Service Time: N/A")
    print(f"   🔗 Has Graph: {info['has_graph']}")
    if info['has_graph']:
        print(f"   📍 Nodes: {info['num_nodes']}")
        print(f"   🔗 Edges: {info['num_edges']}")
    
    # Make predictions
    predictions = predictor_singleton.predict_minutes(test_stops)
    
    print(f"\n🔮 Predictions:")
    for i, (stop, pred) in enumerate(zip(test_stops, predictions)):
        print(f"   📍 {stop['id']}: {pred:.1f} minutes")
        print(f"      Demand: {stop['demand']}, Access: {stop['access_score']:.2f}")
    
    print(f"\n✅ Service time predictor ready!")

if __name__ == "__main__":
    test_service_time_predictor()