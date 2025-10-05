"""
Service Time Model with Knowledge Graph + GNN integration
"""

import torch, json, os
import pandas as pd

ART = "mlartifacts/service_time_gnn.pt"
ART_MLP = "mlartifacts/service_time_mlp.pt"

class ServiceTimePredictor:
    def __init__(self):
        self.mode = None
        if os.path.exists(ART):
            try:
                from torch_geometric.nn import SAGEConv  # ensure pyg present
                # re-create model skeleton
                ckpt = torch.load(ART, map_location="cpu", weights_only=False)
                self.X = ckpt["X"]; self.edge_index = ckpt["edge_index"]; self.y_mean = ckpt["y_mean"]
                class SAGEReg(torch.nn.Module):
                    def __init__(self,in_node=2,in_tab=4, hid=64):
                        super().__init__()
                        from torch_geometric.nn import SAGEConv
                        self.g1=SAGEConv(in_node,hid); self.g2=SAGEConv(hid,hid)
                        self.mlp=torch.nn.Sequential(torch.nn.Linear(hid+in_tab,64), torch.nn.ReLU(), torch.nn.Linear(64,1))
                    def forward(self,x, ei, idxb, xt):
                        h=torch.relu(self.g1(x,ei)); h=torch.relu(self.g2(h,ei))
                        hb=h[idxb]; return self.mlp(torch.cat([hb,xt],1)).squeeze(-1)
                self.model = SAGEReg().eval()
                self.model.load_state_dict(ckpt["state_dict"])
                self.mode = "gnn"
            except Exception as e:
                print(f"GNN model load failed: {e}")
                self.mode = None
        
        if self.mode is None and os.path.exists(ART_MLP):
            try:
                ckpt = torch.load(ART_MLP, map_location="cpu", weights_only=False)
                self.y_mean = ckpt["y_mean"]
                class M(torch.nn.Module):
                    def __init__(self, in_dim=6):
                        super().__init__()
                        self.mlp=torch.nn.Sequential(torch.nn.Linear(in_dim,128), torch.nn.ReLU(),
                                                     torch.nn.Linear(128,64), torch.nn.ReLU(),
                                                     torch.nn.Linear(64,1))
                    def forward(self,x): return self.mlp(x).squeeze(-1)
                self.model = M().eval()
                self.model.load_state_dict(ckpt["state_dict"])
                self.mode = "mlp"
            except Exception as e:
                print(f"MLP model load failed: {e}")
                self.mode = None
        
        if self.mode is None:
            print("No trained model available - using heuristic fallback")
            self.mode = "none"

    def predict_minutes(self, stop_rows):
        """
        stop_rows: list of dicts with keys:
         id, demand, access_score, hour, weekday, node_idx (optional)
        returns: list[float] predicted minutes
        """
        if self.mode == "none":
            # heuristic fallback
            return [max(3.0, 4 + 0.06*r.get("demand",120) + 5*(1-r.get("access_score",0.6))) for r in stop_rows]
        if self.mode == "gnn":
            import numpy as np
            # map ids to node indices (train mapping was position in kg_nodes.csv)
            # for now, expect caller to send node_idx; otherwise default to zeros
            idx = torch.tensor([r.get("node_idx",0) for r in stop_rows], dtype=torch.long)
            xt  = torch.tensor([[r.get("demand",120), r.get("access_score",0.6),
                                 r.get("hour",10), r.get("weekday",2)] for r in stop_rows], dtype=torch.float32)
            with torch.no_grad():
                y = self.model(self.X, self.edge_index, idx, xt)
            return y.clamp_min(3.0).tolist()
        else:
            import numpy as np
            # MLP input = node(demand,access) + tab (demand,access,hour,weekday)
            # caller provides demand/access already; duplicate as node features
            xt  = torch.tensor([[r.get("demand",120), r.get("access_score",0.6),
                                 r.get("demand",120), r.get("access_score",0.6),
                                 r.get("hour",10), r.get("weekday",2)] for r in stop_rows], dtype=torch.float32)
            with torch.no_grad():
                y = self.model(xt)
            return y.clamp_min(3.0).tolist()

predictor_singleton = ServiceTimePredictor()