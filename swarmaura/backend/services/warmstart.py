"""
Warm-start Clusterer for initial route generation
"""

import torch, numpy as np, os, math
ART = "mlartifacts/warmstart_clf.pt"

class WarmStart:
    def __init__(self):
        self.model=None; self.K=None
        if os.path.exists(ART):
            try:
                ck = torch.load(ART, map_location="cpu", weights_only=False)
                self.K = ck["K"]
                class Clf(torch.nn.Module):
                    def __init__(self, in_dim=6, K=ck["K"]):
                        super().__init__()
                        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim,128), torch.nn.ReLU(),
                                                       torch.nn.Linear(128,64), torch.nn.ReLU(),
                                                       torch.nn.Linear(64,K))
                    def forward(self,x): return self.net(x)
                self.model = Clf(); self.model.load_state_dict(ck["state_dict"]); self.model.eval()
                print("✅ Warm-start clusterer model loaded")
            except Exception as e:
                print(f"⚠️  Warm-start clusterer model load failed: {e}")
                self.model = None
        else:
            print("⚠️  No warm-start clusterer found - using KMeans fallback")
            self.model = None
    
    def predict_clusters(self, depot, stops, vehicles):
        K = len(vehicles)
        feats = np.array([[s["lat"], s["lng"], s.get("demand",0), s.get("priority",1), 10, 2] for s in stops], dtype=np.float32)
        if self.model is None or K != self.K:
            # fallback: KMeans on lat/lng/demand (no sklearn to keep deps light)
            # simple farthest-point seeding + 10 Lloyd iterations
            P = feats[:, :3]  # lat,lng,demand
            # seed
            cent = [P[np.random.randint(len(P))]]
            for _ in range(1,K):
                d = np.min([np.linalg.norm(P-c,axis=1) for c in cent], axis=0)
                cent.append(P[np.argmax(d)])
            C = np.stack(cent,0)
            for _ in range(10):
                assign = np.argmin(((P[:,None,:]-C[None,:,:])**2).sum(-1), axis=1)
                for k in range(K):
                    m = (assign==k)
                    if m.any(): C[k]=P[m].mean(0)
            return assign.tolist()
        with torch.no_grad():
            logits = self.model(torch.tensor([[s["lat"], s["lng"], s.get("demand",0), s.get("priority",1), 10, 2] for s in stops], dtype=torch.float32))
            return logits.argmax(1).tolist()

    def build_initial_routes(self, depot, stops, vehicles):
        K = len(vehicles)
        lab = self.predict_clusters(depot, stops, vehicles)
        # nearest-neighbor within each cluster
        routes = [[] for _ in range(K)]
        # start from depot
        for k in range(K):
            idxs = [i for i,l in enumerate(lab) if l==k]
            routes[k] = [0] + [i+1 for i in idxs] + [0]  # 0=depot, 1..N stops order (rough)
        return routes

warmstart_singleton = WarmStart()
