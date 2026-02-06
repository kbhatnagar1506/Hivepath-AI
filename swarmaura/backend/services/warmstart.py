"""
Warm-Start Clusterer Service
Predicts cluster assignments for stops to seed OR-Tools with good initial routes
"""

import torch
import numpy as np
import os
import math
from typing import List, Dict, Any

ART = "mlartifacts/warmstart_clf.pt"

class WarmStart:
    """Warm-start clusterer for routing optimization"""
    
    def __init__(self):
        self.model = None
        self.K = None
        
        if os.path.exists(ART):
            try:
                ck = torch.load(ART, map_location="cpu", weights_only=False)
                self.K = ck["K"]
                self.model = self._create_model(ck["K"])
                self.model.load_state_dict(ck["state_dict"])
                self.model.eval()
                print(f"✅ Loaded warm-start clusterer (K={self.K}, Acc: {ck.get('val_acc', 'N/A'):.3f})")
            except Exception as e:
                print(f"⚠️  Failed to load warm-start clusterer: {e}")
                self.model = None
        else:
            print("⚠️  No warm-start clusterer found - using KMeans fallback")
    
    def _create_model(self, K):
        """Create the clustering classifier architecture"""
        import torch.nn as nn
        
        class ClusteringClassifier(nn.Module):
            def __init__(self, in_dim=6, K=K):
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
        
        return ClusteringClassifier(K=K)
    
    def predict_clusters(self, depot: Dict[str, Any], stops: List[Dict[str, Any]], vehicles: List[Dict[str, Any]]) -> List[int]:
        """
        Predict cluster assignments for stops
        
        Args:
            depot: Depot information
            stops: List of stop dictionaries
            vehicles: List of vehicle dictionaries
        
        Returns:
            List of cluster assignments (0 to K-1) for each stop
        """
        K = len(vehicles)
        
        if not stops:
            return []
        
        # Prepare features: [lat, lng, demand, priority, hour, weekday]
        feats = []
        for s in stops:
            feat = [
                s["lat"],
                s["lng"],
                s.get("demand", 0),
                s.get("priority", 1),
                10,  # Default hour
                2    # Default weekday
            ]
            feats.append(feat)
        
        feats = np.array(feats, dtype=np.float32)
        
        if self.model is None or K != self.K:
            # Fallback: KMeans on lat/lng/demand
            return self._kmeans_fallback(feats, K)
        
        # Use trained model
        with torch.no_grad():
            logits = self.model(torch.tensor(feats, dtype=torch.float32))
            predictions = logits.argmax(1).tolist()
        
        return predictions
    
    def _kmeans_fallback(self, feats: np.ndarray, K: int) -> List[int]:
        """KMeans fallback clustering"""
        P = feats[:, :3]  # lat, lng, demand
        
        # Simple farthest-point seeding
        centroids = [P[np.random.randint(len(P))]]
        
        for _ in range(1, K):
            distances = np.min([np.linalg.norm(P - c, axis=1) for c in centroids], axis=0)
            centroids.append(P[np.argmax(distances)])
        
        centroids = np.stack(centroids, 0)
        
        # Lloyd iterations
        for _ in range(10):
            assignments = np.argmin(((P[:, None, :] - centroids[None, :, :]) ** 2).sum(-1), axis=1)
            
            for k in range(K):
                mask = (assignments == k)
                if mask.any():
                    centroids[k] = P[mask].mean(0)
        
        return assignments.tolist()
    
    def build_initial_routes(self, depot: Dict[str, Any], stops: List[Dict[str, Any]], vehicles: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Build initial routes from cluster assignments
        
        Args:
            depot: Depot information
            stops: List of stop dictionaries
            vehicles: List of vehicle dictionaries
        
        Returns:
            List of initial routes (each route is a list of node indices)
        """
        K = len(vehicles)
        
        if not stops:
            return [[0] for _ in range(K)]  # Empty routes
        
        # Get cluster assignments
        labels = self.predict_clusters(depot, stops, vehicles)
        
        # Build routes: nearest-neighbor within each cluster
        routes = [[] for _ in range(K)]
        
        # Start from depot (node 0)
        for k in range(K):
            # Get stops assigned to this cluster
            cluster_stops = [i for i, label in enumerate(labels) if label == k]
            
            if not cluster_stops:
                # Empty cluster - just depot
                routes[k] = [0]
                continue
            
            # Simple nearest-neighbor within cluster
            route = [0]  # Start at depot
            remaining = cluster_stops.copy()
            current = 0  # Depot
            
            while remaining:
                # Find nearest stop in cluster
                distances = []
                for stop_idx in remaining:
                    # Calculate distance (simplified)
                    stop = stops[stop_idx]
                    if current == 0:  # From depot
                        dist = math.sqrt((depot["lat"] - stop["lat"])**2 + (depot["lng"] - stop["lng"])**2)
                    else:  # From another stop
                        prev_stop = stops[current - 1]  # Convert to 0-based index
                        dist = math.sqrt((prev_stop["lat"] - stop["lat"])**2 + (prev_stop["lng"] - stop["lng"])**2)
                    distances.append(dist)
                
                # Add nearest stop
                nearest_idx = remaining[np.argmin(distances)]
                route.append(nearest_idx + 1)  # Convert to 1-based (depot=0, stops=1..N)
                remaining.remove(nearest_idx)
                current = nearest_idx + 1
            
            route.append(0)  # Return to depot
            routes[k] = route
        
        return routes
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "loaded": self.model is not None,
            "K": self.K,
            "model_type": "mlp" if self.model else "kmeans_fallback"
        }

# Global singleton instance
warmstart_singleton = WarmStart()

def test_warmstart():
    """Test the warm-start clusterer"""
    print("🧪 Testing Warm-Start Clusterer")
    print("=" * 35)
    
    # Test data
    depot = {
        "id": "D",
        "lat": 42.3601,
        "lng": -71.0589,
        "name": "Main Depot"
    }
    
    stops = [
        {"id": "S_A", "lat": 42.37, "lng": -71.05, "demand": 150, "priority": 1},
        {"id": "S_B", "lat": 42.34, "lng": -71.10, "demand": 140, "priority": 2},
        {"id": "S_C", "lat": 42.39, "lng": -71.02, "demand": 160, "priority": 1},
        {"id": "S_D", "lat": 42.33, "lng": -71.06, "demand": 130, "priority": 3},
        {"id": "S_E", "lat": 42.41, "lng": -71.03, "demand": 145, "priority": 2}
    ]
    
    vehicles = [
        {"id": "truck_1", "capacity": 500},
        {"id": "truck_2", "capacity": 500}
    ]
    
    # Get model info
    info = warmstart_singleton.get_model_info()
    print(f"📊 Model Info:")
    print(f"   🎯 Loaded: {info['loaded']}")
    print(f"   🎯 K: {info['K']}")
    print(f"   🎯 Type: {info['model_type']}")
    
    # Test cluster prediction
    print(f"\n🔮 Testing cluster prediction...")
    labels = warmstart_singleton.predict_clusters(depot, stops, vehicles)
    print(f"   📊 Cluster assignments: {labels}")
    
    # Show cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   📊 Cluster distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # Test route building
    print(f"\n🚛 Testing route building...")
    routes = warmstart_singleton.build_initial_routes(depot, stops, vehicles)
    
    print(f"   📊 Generated {len(routes)} routes:")
    for i, route in enumerate(routes):
        print(f"      Route {i+1}: {route}")
    
    # Show route details
    print(f"\n📋 Route Details:")
    for i, route in enumerate(routes):
        if len(route) > 2:  # More than just depot->depot
            stops_in_route = [stops[idx-1]["id"] for idx in route[1:-1]]  # Skip depot at start/end
            print(f"   🚛 Route {i+1}: Depot → {' → '.join(stops_in_route)} → Depot")
        else:
            print(f"   🚛 Route {i+1}: Empty (Depot → Depot)")
    
    print(f"\n✅ Warm-start clusterer test complete!")

if __name__ == "__main__":
    test_warmstart()
