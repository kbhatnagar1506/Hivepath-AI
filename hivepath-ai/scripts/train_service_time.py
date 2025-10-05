#!/usr/bin/env python3
import pandas as pd, numpy as np, joblib, pathlib, argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

def load_df(path:str):
    df = pd.read_csv(path)
    # Label
    if "service_minutes" not in df.columns:
        df["service_minutes"] = (df["departed_min"] - df["arrived_min"]).clip(lower=1, upper=120)
    # Basic cleaning
    df["priority"] = df["priority"].fillna(1).astype(int)
    df["blocked_flag"] = df["blocked_flag"].fillna(False).astype(int)
    df["walk_m"] = df["walk_m"].fillna(0.0)
    # Feature hints if windows exist
    def parse_hhmm(s):
        try:
            h, m, _ = map(int, str(s).split(":"))
            return h*60+m
        except:
            return np.nan
    df["win_start_min"] = df["window_start"].apply(parse_hhmm) if "window_start" in df else np.nan
    df["win_end_min"]   = df["window_end"].apply(parse_hhmm)   if "window_end" in df else np.nan
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/service_actuals.csv")
    ap.add_argument("--out", default="models/service_time.joblib")
    args = ap.parse_args()

    df = load_df(args.csv)
    # Train/valid split
    msk = np.random.RandomState(42).rand(len(df)) < 0.85
    tr, va = df[msk], df[~msk]

    num = ["lat","lng","demand","walk_m","win_start_min","win_end_min","hod","dow"]
    cat = ["priority","blocked_flag"]
    X_cols = num + cat
    y = "service_minutes"

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

    rf = RandomForestRegressor(
        n_estimators=250, max_depth=12, random_state=42, n_jobs=-1
    )
    pipe = Pipeline([("pre", pre), ("rf", rf)])

    pipe.fit(tr[X_cols], tr[y])
    pred = pipe.predict(va[X_cols])
    mae = mean_absolute_error(va[y], pred)
    print(f"Validation MAE (minutes): {mae:.2f}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "features": X_cols}, args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()



