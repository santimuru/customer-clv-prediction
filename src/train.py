"""
train.py — Trains a Customer Lifetime Value (CLV) prediction model.

Pipeline:
  1. Downloads the UCI Online Retail dataset (~23MB Excel)
  2. Cleans and transforms transaction data into RFM format
  3. Fits BG/NBD model (purchase frequency prediction)
  4. Fits Gamma-Gamma model (monetary value prediction)
  5. Computes CLV per customer at 6 and 12 month horizons
  6. Trains a RandomForest regressor as a comparison baseline
  7. Saves models and metadata to models/

Usage:
    python src/train.py
"""

import os
import sys
import pickle
import warnings
import joblib
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE_DIR, "data", "online_retail.xlsx")
DATA_PATH_CSV = os.path.join(BASE_DIR, "data", "online_retail.csv")
MODEL_BGN  = os.path.join(BASE_DIR, "models", "bgn_model.pkl")
MODEL_GG   = os.path.join(BASE_DIR, "models", "gg_model.pkl")
MODEL_RF   = os.path.join(BASE_DIR, "models", "rf_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")

# Primary: Databricks GitHub mirror (reliable)
# Fallback: UCI endpoints
DATASET_URLS = [
    ("csv",  "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"),
    ("zip",  "https://archive.ics.uci.edu/static/public/352/online+retail.zip"),
    ("xlsx", "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"),
]

# ─── RFM Segment labels ───────────────────────────────────────────────────────
def assign_segment(row):
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    rfm = r + f + m
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "Recent Customers"
    elif r >= 3 and f <= 2:
        return "Promising"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f >= 4:
        return "Can't Lose Them"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    else:
        return "Need Attention"


def download_data():
    if os.path.exists(DATA_PATH):
        print(f"[OK] Dataset already exists at {DATA_PATH}")
        return pd.read_excel(DATA_PATH)
    if os.path.exists(DATA_PATH_CSV):
        print(f"[OK] Dataset (CSV) already exists at {DATA_PATH_CSV}")
        return pd.read_csv(DATA_PATH_CSV, encoding="latin1")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    for fmt, url in DATASET_URLS:
        print(f"[>>] Trying {fmt.upper()} from {url[:60]}...")
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
        except Exception as e:
            print(f"    [!] Failed: {e}")
            continue

        if fmt == "zip":
            import zipfile
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                xlsx_names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
                if not xlsx_names:
                    print("    [!] No xlsx found in zip")
                    continue
                with z.open(xlsx_names[0]) as f:
                    data = f.read()
            with open(DATA_PATH, "wb") as f:
                f.write(data)
            print(f"[OK] Saved to {DATA_PATH}")
            return pd.read_excel(BytesIO(data))

        elif fmt == "xlsx":
            with open(DATA_PATH, "wb") as f:
                f.write(r.content)
            print(f"[OK] Saved to {DATA_PATH}")
            return pd.read_excel(BytesIO(r.content))

        elif fmt == "csv":
            with open(DATA_PATH_CSV, "wb") as f:
                f.write(r.content)
            print(f"[OK] Saved to {DATA_PATH_CSV}")
            return pd.read_csv(BytesIO(r.content), encoding="latin1")

    raise RuntimeError("Could not download dataset from any source.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Remove rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    # Remove cancellations (InvoiceNo starting with C)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    # Keep only positive quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    # Compute revenue per line
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    # Parse date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def build_rfm(df: pd.DataFrame, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm_summary = summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=snapshot_date,
        freq="D",
    )
    # Only customers with at least 2 purchases (required for BG/NBD)
    rfm = rfm_summary[rfm_summary["frequency"] > 0].copy()

    # RFM scores (quintiles 1-5)
    rfm["r_score"] = pd.qcut(rfm["recency"],    5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary_value"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    return rfm, snapshot_date


def fit_bgn(rfm: pd.DataFrame):
    print("[~] Fitting BG/NBD model...")
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    return bgf


def fit_gg(rfm: pd.DataFrame):
    print("[~] Fitting Gamma-Gamma model...")
    gg = GammaGammaFitter(penalizer_coef=0.001)
    # Gamma-Gamma requires frequency > 0 (already filtered)
    gg.fit(rfm["frequency"], rfm["monetary_value"])
    return gg


def compute_clv(rfm, bgf, gg, months=12, monthly_discount=0.01):
    clv = gg.customer_lifetime_value(
        bgf,
        rfm["frequency"],
        rfm["recency"],
        rfm["T"],
        rfm["monetary_value"],
        time=months,
        freq="D",
        discount_rate=monthly_discount,
    )
    return clv


def train_rf_baseline(rfm: pd.DataFrame, clv_12m: pd.Series):
    features = ["frequency", "recency", "T", "monetary_value", "r_score", "f_score", "m_score"]
    X = rfm[features]
    y = clv_12m.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"    RF Baseline — MAE: £{mae:.2f}  R2: {r2:.4f}")
    return pipe, {"mae": round(mae, 2), "r2": round(r2, 4)}


def main():
    # 1. Data
    df_raw = download_data()
    print(f"[i] Raw shape: {df_raw.shape}")
    df = clean_data(df_raw)
    print(f"[i] Clean shape: {df.shape}  |  Customers: {df['CustomerID'].nunique()}")

    # 2. RFM
    rfm, snapshot_date = build_rfm(df)
    print(f"[i] RFM customers (freq > 0): {len(rfm)}")
    print(f"[i] Snapshot date: {snapshot_date.date()}")

    # 3. BG/NBD + Gamma-Gamma
    bgf = fit_bgn(rfm)
    gg  = fit_gg(rfm)

    # 4. CLV projections
    print("[~] Computing CLV projections...")
    rfm["clv_6m"]  = np.array(compute_clv(rfm, bgf, gg, months=6))
    rfm["clv_12m"] = np.array(compute_clv(rfm, bgf, gg, months=12))
    rfm["p_alive"]  = np.array(bgf.conditional_probability_alive(
        rfm["frequency"], rfm["recency"], rfm["T"]
    ))
    rfm["expected_purchases_90d"] = np.array(
        bgf.conditional_expected_number_of_purchases_up_to_time(
            90, rfm["frequency"], rfm["recency"], rfm["T"]
        )
    )

    # 5. RF baseline
    print("[~] Training RF baseline...")
    rf_model, rf_metrics = train_rf_baseline(rfm, rfm["clv_12m"])

    # 6. Cohort analysis
    df["cohort"] = df.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
    df["invoice_period"] = df["InvoiceDate"].dt.to_period("M")
    cohort_data = (
        df.groupby(["cohort", "invoice_period"])["CustomerID"]
        .nunique()
        .reset_index()
    )
    cohort_data["period_number"] = (
        cohort_data["invoice_period"] - cohort_data["cohort"]
    ).apply(lambda x: x.n)
    cohort_pivot = cohort_data.pivot_table(
        index="cohort", columns="period_number", values="CustomerID"
    )
    cohort_retention = cohort_pivot.div(cohort_pivot[0], axis=0).round(3)

    # 7. Country revenue breakdown
    country_rev = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # 8. Monthly revenue trend
    monthly_rev = (
        df.set_index("InvoiceDate")
        .resample("ME")["Revenue"]
        .sum()
        .reset_index()
    )
    monthly_rev.columns = ["date", "revenue"]

    # 9. Save
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    # lifetimes models can't be pickled (contain internal lambdas)
    # — save fitted params instead and reconstruct on load
    joblib.dump(rf_model, MODEL_RF)

    segment_stats = (
        rfm.groupby("segment")
        .agg(
            customers=("clv_12m", "count"),
            avg_clv_12m=("clv_12m", "mean"),
            total_clv_12m=("clv_12m", "sum"),
            avg_p_alive=("p_alive", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary_value", "mean"),
        )
        .round(2)
        .reset_index()
    )

    meta = {
        "snapshot_date": snapshot_date,
        "n_customers_raw": df["CustomerID"].nunique(),
        "n_customers_rfm": len(rfm),
        "n_transactions": len(df),
        "rfm": rfm.reset_index(),
        "segment_stats": segment_stats,
        "cohort_retention": cohort_retention,
        "country_revenue": country_rev,
        "monthly_revenue": monthly_rev,
        "rf_metrics": rf_metrics,
        "total_clv_6m":  round(rfm["clv_6m"].sum(), 2),
        "total_clv_12m": round(rfm["clv_12m"].sum(), 2),
        "median_clv_12m": round(rfm["clv_12m"].median(), 2),
        "avg_p_alive": round(rfm["p_alive"].mean(), 4),
        # lifetimes model params (serialized as plain dicts — models reconstructed on load)
        "bgf_params": dict(bgf.params_),
        "gg_params":  dict(gg.params_),
    }
    joblib.dump(meta, META_PATH)

    print(f"\n[OK] Models saved.")
    print(f"[OK] Metadata saved to {META_PATH}")
    print(f"\n  Customers analyzed : {len(rfm):,}")
    print(f"  Projected CLV 12m  : £{meta['total_clv_12m']:,.0f}")
    print(f"  Median CLV/customer: £{meta['median_clv_12m']:,.2f}")
    print(f"  Avg P(alive)       : {meta['avg_p_alive']:.1%}")

    print("\n[Segment breakdown:]")
    print(segment_stats.to_string(index=False))


if __name__ == "__main__":
    main()
