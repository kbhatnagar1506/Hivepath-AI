#!/usr/bin/env python3
import pandas as pd, numpy as np, joblib, argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/plan_edges.csv")
    ap.add_argument("--out", default="models/warmstart_edge_clf.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Minimal features
    num = ["dist_km","curr_demand","next_demand","next_priority","window_slack_min"]
    X, y = df[num], df["label"].astype(int)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
    ])
    pipe.fit(X, y)
    auc = roc_auc_score(y, pipe.predict_proba(X)[:,1])
    print(f"Train AUC: {auc:.3f}")
    joblib.dump({"model": pipe, "features": num}, args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
