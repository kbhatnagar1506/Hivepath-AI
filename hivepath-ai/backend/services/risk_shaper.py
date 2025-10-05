"""
Risk Shaper for Edge-level Risk/Cost adjustment
"""

import torch, numpy as np, os

ART = "mlartifacts/risk_edge.pt"

class RiskShaper:
    def __init__(self):
        if os.path.exists(ART):
            try:
                ck = torch.load(ART, map_location="cpu", weights_only=False)
                self.stops = ck["stops"]; self.sid = {s:i for i,s in enumerate(self.stops)}
                self.model = self._mk(); self.model.load_state_dict(ck["state_dict"]); self.model.eval()
                print("✅ Risk shaper model loaded")
            except Exception as e:
                print(f"⚠️  Risk shaper model load failed: {e}")
                self.model = None
        else:
            print("⚠️  No risk shaper model found - using identity multipliers")
            self.model = None
    
    def _mk(self):
        import torch.nn as nn
        class PairMLP(nn.Module):
            def __init__(self, in_node=3, in_tab=10, hid=64):
                super().__init__()
                self.enc_i = nn.Sequential(nn.Linear(in_node, hid), nn.ReLU())
                self.enc_j = nn.Sequential(nn.Linear(in_node, hid), nn.ReLU())
                self.head  = nn.Sequential(nn.Linear(2*hid+in_tab,64), nn.ReLU(), nn.Linear(64,1))
            def forward(self, X, i, j, tab):
                import torch.nn.functional as F
                hi = self.enc_i(X[i]); hj = self.enc_j(X[j])
                z  = torch.cat([hi,hj,tab], dim=1)
                return F.softplus(self.head(z)).squeeze(-1)
        return PairMLP()
    
    def shape(self, stops_order, osrm_min_matrix, hour:int, weekday:int, features_by_stop:dict):
        """
        stops_order: list of stop ids in same order as matrix (0=depot, 1..N=stops)
        features_by_stop[id] = {risk, light, cong}
        returns: multiplier matrix M (N+1 x N+1), with zeros on diag
        """
        n = len(stops_order)
        M = np.zeros((n,n), dtype=np.float32)
        if self.model is None: return M
        
        # Build node feature matrix X (depot -> average stats)
        def feat(sid):
            f = features_by_stop.get(sid, {"risk":0.5, "light":0.5, "cong":0.5})
            return [f["risk"], f["light"], f["cong"]]
        X = torch.tensor([feat(s) for s in stops_order], dtype=torch.float32)
        pairs = []
        tabs  = []
        is_idx, js_idx = [], []
        for i in range(n):
            for j in range(n):
                if i==j: continue
                base = osrm_min_matrix[i][j]
                if base <= 0 or base >= 1e6: continue
                fi = features_by_stop.get(stops_order[i], {"risk":0.5,"light":0.5,"cong":0.5})
                fj = features_by_stop.get(stops_order[j], {"risk":0.5,"light":0.5,"cong":0.5})
                tab = [base, weekday, hour, fi["risk"], fj["risk"], fi["light"], fj["light"], fi["cong"], fj["cong"], 0]
                tabs.append(tab); is_idx.append(i); js_idx.append(j)
        if not tabs: return M
        TAB = torch.tensor(np.array(tabs, dtype=np.float32))
        with torch.no_grad():
            pred = self.model(X, torch.tensor(is_idx), torch.tensor(js_idx), TAB).numpy()
        for k,(i,j) in enumerate(zip(is_idx, js_idx)):
            M[i,j] = float(pred[k])
        return M

risk_shaper_singleton = RiskShaper()
