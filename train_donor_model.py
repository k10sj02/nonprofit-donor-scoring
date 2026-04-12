"""
train_donor_model.py

Run this script ONCE locally to train and save the donor propensity model artifact.
The Streamlit app loads this file instead of training at runtime — no training on Render.

Usage:
    uv run python train_donor_model.py
"""

import hashlib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42
BASE         = Path(__file__).parent
OUTPUT_PATH  = BASE / "outputs" / "donor_model.joblib"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_donor_features(transactions: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    g = transactions.groupby("donor_id")

    summary = g.agg(
        donation_count =("donation_id",    "count"),
        total_donated  =("donation_amount", "sum"),
        avg_donation   =("donation_amount", "mean"),
        last_donation  =("donation_date",   "max"),
        first_donation =("donation_date",   "min"),
    ).reset_index()

    summary["recency_days"]            = (ref_date - summary["last_donation"]).dt.days
    summary["months_since_last"]       = summary["recency_days"] / 30.44
    summary["days_since_first"]        = (ref_date - summary["first_donation"]).dt.days
    summary["amount_log"]              = np.log1p(summary["total_donated"])
    summary["avg_donation_log"]        = np.log1p(summary["avg_donation"])
    active_days                         = summary["days_since_first"].replace(0, 1)
    summary["giving_frequency_ratio"]  = summary["donation_count"] / active_days
    summary["first_gift_missing_flag"] = summary["first_donation"].isna().astype(int)
    summary["never_donated_flag"]      = 0

    lifetime = transactions.groupby("donor_id").agg(
        gift_100_lifetime=("donation_amount", lambda x: int((x >= 100).any())),
    ).reset_index()
    summary = summary.merge(lifetime, on="donor_id", how="left")

    if "newsletter_opt_in" in transactions.columns:
        nl = (transactions.sort_values("donation_date")
              .groupby("donor_id")["newsletter_opt_in"].last().reset_index())
        nl["newsletter_opt_in"] = nl["newsletter_opt_in"].astype(int)
        summary = summary.merge(nl, on="donor_id", how="left")
        summary["newsletter_opt_in"] = summary["newsletter_opt_in"].fillna(0).astype(int)

    if "referral_channel" in transactions.columns:
        ref = transactions.groupby("donor_id")["referral_channel"].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index()
        dummies = pd.get_dummies(ref["referral_channel"], prefix="ref").astype(int)
        ref = pd.concat([ref[["donor_id"]], dummies], axis=1)
        summary = summary.merge(ref, on="donor_id", how="left")
        for c in [col for col in summary.columns if col.startswith("ref_")]:
            summary[c] = summary[c].fillna(0).astype(int)

    if "age_group" in transactions.columns:
        age = transactions.groupby("donor_id")["age_group"].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index()
        dummies = pd.get_dummies(age["age_group"], prefix="age").astype(int)
        age = pd.concat([age[["donor_id"]], dummies], axis=1)
        summary = summary.merge(age, on="donor_id", how="left")
        for c in [col for col in summary.columns if col.startswith("age_")]:
            summary[c] = summary[c].fillna(0).astype(int)

    return summary


def recall_at_k(y_true, y_score, k):
    idx = np.argsort(-np.array(y_score))[:max(1, int(len(y_true) * k))]
    return float(np.array(y_true)[idx].sum() / max(1, np.array(y_true).sum()))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔄 Loading data...")
    data_path = BASE / "outputs" / "donations_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"donations_clean.csv not found at {data_path}. "
            "Run the analysis notebook first."
        )

    df = pd.read_csv(data_path)
    df["donation_date"] = pd.to_datetime(df["donation_date"])

    # Time-based split
    cutoff_date = df["donation_date"].quantile(0.8)
    train_df    = df[df["donation_date"] <= cutoff_date]
    future_df   = df[df["donation_date"] >  cutoff_date]

    print(f"   Training window: {train_df['donation_date'].min().date()} → {cutoff_date.date()}")
    print(f"   Future window:   {cutoff_date.date()} → {future_df['donation_date'].max().date()}")

    # Feature engineering
    print("\n🔧 Building features...")
    donor_summary = build_donor_features(train_df, cutoff_date)
    future_donors = future_df["donor_id"].unique()
    donor_summary["donated_again"] = donor_summary["donor_id"].isin(future_donors).astype(int)

    print(f"   Donors: {len(donor_summary):,} | Retention rate: {donor_summary['donated_again'].mean():.1%}")

    BASE_FEATURES = [
        "recency_days", "donation_count", "amount_log", "avg_donation_log",
        "months_since_last", "days_since_first", "giving_frequency_ratio",
        "gift_100_lifetime", "first_gift_missing_flag", "never_donated_flag",
        "newsletter_opt_in",
    ]
    EXTRA_FEATURES = [c for c in donor_summary.columns
                      if c.startswith("ref_") or c.startswith("age_")]
    MODEL_FEATURES = [f for f in BASE_FEATURES + EXTRA_FEATURES if f in donor_summary.columns]

    X = donor_summary[MODEL_FEATURES].fillna(0)
    y = donor_summary["donated_again"]

    # Time-based train/test split for evaluation
    split_cutoff    = donor_summary["last_donation"].quantile(0.8)
    train_mask      = donor_summary["last_donation"] <= split_cutoff
    test_mask       = donor_summary["last_donation"] >  split_cutoff
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_test) < 50 or y_test.nunique() < 2:
        X_train, X_test = X, X
        y_train, y_test = y, y

    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train model
    print("\n🌲 Training RandomForest (n_estimators=300)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Cross-validated ROC-AUC
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_aucs = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    roc_auc = float(cv_aucs.mean())
    print(f"   CV ROC-AUC: {roc_auc:.4f} (±{cv_aucs.std():.4f})")

    # Final fit on training window
    rf.fit(X_train, y_train)
    p_test = rf.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, p_test))
    r10    = recall_at_k(y_test.values, p_test, 0.10)
    r20    = recall_at_k(y_test.values, p_test, 0.20)
    r33    = recall_at_k(y_test.values, p_test, 0.33)

    print(f"   PR-AUC:        {pr_auc:.4f}")
    print(f"   Recall @10%:   {r10:.1%}")
    print(f"   Recall @20%:   {r20:.1%}")
    print(f"   Recall @33%:   {r33:.1%}")

    # Pre-compute scores and segments
    scores        = rf.predict_proba(X)[:, 1]
    q80, q60, q40 = np.quantile(scores, [0.80, 0.60, 0.40])

    # Permutation importance (subsampled)
    print("\n📊 Computing permutation importance...")
    rng   = np.random.RandomState(RANDOM_STATE)
    idx   = rng.choice(len(X_test), min(300, len(X_test)), replace=False)
    perm  = permutation_importance(
        rf, X_test.iloc[idx], y_test.iloc[idx],
        n_repeats=5, random_state=RANDOM_STATE, scoring="average_precision",
    )
    imp_df = pd.DataFrame({
        "feature":    X_test.columns,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False).head(15).reset_index(drop=True)

    # Data hash for cache validation in Streamlit
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(donor_summary, index=True).values
    ).hexdigest()

    # Save artifact
    artifact = dict(
        model          = rf,
        roc_auc        = roc_auc,
        pr_auc         = pr_auc,
        recall_top10   = r10,
        recall_top20   = r20,
        scores         = scores.tolist(),
        segments       = [
            "High"     if s >= q80 else
            "Medium"   if s >= q60 else
            "Low"      if s >= q40 else
            "Very Low"
            for s in scores
        ],
        imp_df         = imp_df,
        model_features = MODEL_FEATURES,
        donor_summary  = donor_summary,
        test_mask      = test_mask.values,
        data_hash      = data_hash,
        cutoff_date    = str(cutoff_date),
    )

    joblib.dump(artifact, OUTPUT_PATH)
    print(f"\n✅ Model artifact saved → {OUTPUT_PATH}")
    print(f"   File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
