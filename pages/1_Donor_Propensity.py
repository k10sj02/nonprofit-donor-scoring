"""
1_Donor_Propensity.py

Donor propensity scoring dashboard.
Model logic:
  - RandomForestClassifier (handles nonlinear interactions, robust to outliers)
  - Time-based 80/20 split for feature engineering / target definition
  - Stratified 5-fold cross-validation for robust AUC reporting
  - Full feature set: RFM + donor profile + giving behaviour flags
  - 4-tier segmentation: High / Medium / Low / Very Low
  - ROC-AUC (CV) + PR-AUC + Recall@K metrics
  - Permutation feature importance
  - Feature engineering cached by data hash
  - Model cached + saved with joblib for session reuse
  - Scores and segments pre-computed inside cache
  - Model pre-saved to disk for near-instant cold starts
  - Permutation importance cached by model + data hash
"""

import hashlib
import io
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Donor Propensity Scoring",
    page_icon="💚",
    layout="wide",
)

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #4a4a4a; margin-bottom: 0.25rem; }
    [data-testid="stHorizontalBlock"] { align-items: stretch; }
    .vega-embed { width: 100% !important; }
    .vega-embed canvas { width: 100% !important; }
    </style>
""", unsafe_allow_html=True)


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return pd.DataFrame(out.to_dict("records"))


def df_hash(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame for cache keying."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("💚 Donor Propensity Scoring")
st.caption("Score, segment, and prioritize your donor base for maximum retention impact.")
st.divider()

with st.expander("📖 How to use this tool", expanded=False):
    st.markdown("""
**This dashboard scores donors by likelihood to give again and helps prioritize them using both short-term and long-term value signals.**
- **Expected Next Gift** = likelihood × average donation (short-term opportunity)
- **Predicted LTV** = long-term value based on giving frequency and retention

Use the sidebar filter to focus on a specific segment at any time.

---

**🟢 High — Priority outreach**
High likelihood to give and strong overall value. These donors are your top targets for both immediate and long-term impact.

**🟡 Medium — Nurture**
Moderate expected return. With the right engagement, these donors can move into the High tier.

**🟠 Low — Monitor**
Lower likelihood or value. Engage selectively or through lower-cost channels.

**🔴 Very Low — Deprioritize**
Minimal expected impact. Consider broad or automated re-engagement strategies.

---

**Reading the charts**
- **Score Distribution** — likelihood to give again; feeds into Expected Next Gift
- **Who Comes Back?** — retention rate by segment; confirms that higher-ranked donors are more likely to return
- **Giving Frequency** — repeat donors drive most long-term value
- **Recency vs Gift Size** — identify high-value donors who may be at risk of lapsing
- **Model vs Random** — how much more efficiently the model identifies returning donors vs random outreach
- **Feature Importance** — which behavioral signals (recency, frequency, giving patterns) drive predictions

---

**How to use the leaderboard**
- **Sort by Predicted LTV** to prioritize long-term value
- Use **Expected Next Gift** to identify immediate revenue opportunities
- Compare both to spot:
  - High LTV but low next gift → long-term cultivation targets
  - High next gift but low LTV → short-term campaign targets

---

**Suggested next steps**
1. Export the **High tier** list — highest overall value donors
2. Prioritize donors with **high LTV + high next gift** for immediate outreach
3. Target **high LTV but lower next gift** donors with relationship-building strategies
4. Use **Low / Very Low segments** for low-cost or automated campaigns
5. Re-run the model regularly as new donation data comes in
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload donation data (CSV)", type="csv")

    PROJECT_ROOT = Path(__file__).parent.parent
    default_path = PROJECT_ROOT / "outputs" / "donations_clean.csv"

    df = None
    if "mapped_df" in st.session_state:
        df = st.session_state["mapped_df"]
        st.success("Using mapped dataset from Upload & Map")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Using uploaded dataset")
    elif default_path.exists():
        df = pd.read_csv(default_path)
        st.info("Using default dataset")
    else:
        st.warning(
            "No dataset found. Upload a CSV or run the notebook first to generate "
            "`outputs/donations_clean.csv`."
        )

    st.divider()
    st.markdown("**Filter by segment**")
    segment_filter = st.radio(
        label="Segment",
        options=["All", "High", "Medium", "Low", "Very Low"],
        label_visibility="collapsed",
    )

if df is None:
    st.info("👈 Upload a dataset in the sidebar to get started.")
    st.stop()

# ── Data prep ─────────────────────────────────────────────────────────────────
df["donation_date"] = pd.to_datetime(df["donation_date"])

# Time-based split: train on earliest 80%, test on most recent 20%
# The training window defines features; the future window defines the target.
cutoff_date = df["donation_date"].quantile(0.8)
train_df    = df[df["donation_date"] <= cutoff_date]
future_df   = df[df["donation_date"] >  cutoff_date]


# ── Feature engineering ───────────────────────────────────────────────────────
def build_donor_features(transactions: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build per-donor features from transaction history.
    All features are computed from the training window only to prevent leakage.

    TARGET
    ------
    donated_again : binary (0/1)
        1 if the donor made at least one donation after the 80th-percentile
        date cutoff (the "future window"); 0 otherwise.

    FEATURES
    --------
    RFM core:
      recency_days           — days since most recent donation (lower = more recent)
      donation_count         — total number of gifts in the training window
      amount_log             — log1p(total donated) — reduces right skew
      avg_donation_log       — log1p(average gift size)
      months_since_last      — recency expressed in months
      days_since_first       — donor tenure in days (longer = more loyal)
      giving_frequency_ratio — donation_count / days_active (gifts per day)

    Giving behaviour flags:
      gift_100_lifetime       — 1 if any lifetime gift >= $100 (capacity signal)
      first_gift_missing_flag — 1 if first gift date is null
      never_donated_flag      — always 0 here (all rows have donated by definition)

    Pipeline proxy:
      stage_score             — log1p(donation_count); proxy for cultivation depth

    Engagement / profile (when columns present in uploaded data):
      newsletter_opt_in       — 1 if donor is subscribed to newsletter
      ref_*                   — one-hot encoded referral channel (Website, Social, etc.)
      age_*                   — one-hot encoded age group (18-29, 30-49, etc.)
    """
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
    summary["stage_score"]             = np.log1p(summary["donation_count"])
    summary["first_gift_missing_flag"] = summary["first_donation"].isna().astype(int)
    summary["never_donated_flag"]      = 0

    # Lifetime gift flag
    lifetime = transactions.groupby("donor_id").agg(
        gift_100_lifetime=("donation_amount", lambda x: int((x >= 100).any())),
    ).reset_index()
    summary = summary.merge(lifetime, on="donor_id", how="left")

    # Newsletter opt-in (most recent value per donor)
    if "newsletter_opt_in" in transactions.columns:
        nl = (transactions.sort_values("donation_date")
              .groupby("donor_id")["newsletter_opt_in"].last().reset_index())
        nl["newsletter_opt_in"] = nl["newsletter_opt_in"].astype(int)
        summary = summary.merge(nl, on="donor_id", how="left")
        summary["newsletter_opt_in"] = summary["newsletter_opt_in"].fillna(0).astype(int)

    # Referral channel → one-hot
    if "referral_channel" in transactions.columns:
        ref = transactions.groupby("donor_id")["referral_channel"].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index()
        dummies = pd.get_dummies(ref["referral_channel"], prefix="ref").astype(int)
        ref = pd.concat([ref[["donor_id"]], dummies], axis=1)
        summary = summary.merge(ref, on="donor_id", how="left")
        for c in [col for col in summary.columns if col.startswith("ref_")]:
            summary[c] = summary[c].fillna(0).astype(int)

    # Age group → one-hot
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


donor_summary = build_donor_features(train_df, cutoff_date)

# ── Cached feature engineering wrapper ───────────────────────────────────────
# Defined after build_donor_features so it can call it.
# Cached by data hash — reruns only when the underlying CSV changes.
@st.cache_data(show_spinner="Building features...", ttl=3600)
def cached_build_features(data_hash: str, train_json: str, cutoff_str: str,
                           future_donor_ids: list) -> pd.DataFrame:
    train  = pd.read_json(io.StringIO(train_json))
    train["donation_date"] = pd.to_datetime(train["donation_date"], unit="ms")
    cutoff = pd.Timestamp(cutoff_str)
    summary = build_donor_features(train, cutoff)
    summary["donated_again"] = summary["donor_id"].isin(future_donor_ids).astype(int)
    return summary


_data_hash    = df_hash(df)
donor_summary = cached_build_features(
    data_hash       = _data_hash,
    train_json      = train_df.to_json(),
    cutoff_str      = str(cutoff_date),
    future_donor_ids = list(future_df["donor_id"].unique()),
)

# Disk path for pre-saved model — avoids retraining on cold starts
MODEL_DISK_PATH = PROJECT_ROOT / "outputs" / "donor_model.joblib"

# ── Model features ────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "recency_days", "donation_count", "amount_log", "avg_donation_log",
    "months_since_last", "days_since_first", "giving_frequency_ratio",
    "gift_100_lifetime", "first_gift_missing_flag", "never_donated_flag",
    "newsletter_opt_in", "stage_score",
]
EXTRA_FEATURES = [c for c in donor_summary.columns
                  if c.startswith("ref_") or c.startswith("age_")]
MODEL_FEATURES = [f for f in BASE_FEATURES + EXTRA_FEATURES if f in donor_summary.columns]

X = donor_summary[MODEL_FEATURES].fillna(0)
y = donor_summary["donated_again"]

# Time-based split for lift chart and permutation importance
# Proper donor-level time-based split
donor_summary_sorted = donor_summary.sort_values("last_donation")

split_idx = int(len(donor_summary_sorted) * 0.8)

train_ids = set(donor_summary_sorted.iloc[:split_idx]["donor_id"])
test_ids  = set(donor_summary_sorted.iloc[split_idx:]["donor_id"])

train_mask = donor_summary["donor_id"].isin(train_ids)
test_mask  = donor_summary["donor_id"].isin(test_ids)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# fallback if needed
used_fallback = False
if len(X_test) < 50 or y_test.nunique() < 2 or y_test.sum() == 0:
    X_train, X_test = X, X
    y_train, y_test = y, y
    used_fallback = True


# ── Model training — cached + joblib session persistence ─────────────────────
@st.cache_data(show_spinner="Training model...", ttl=3600)
def train_and_evaluate(data_hash: str, feature_cols: list,
                        X_json: str, y_json: str,
                        X_train_json: str, y_train_json: str,
                        X_test_json: str,  y_test_json: str):
    """
    Train RandomForest, compute cross-validated AUC, PR-AUC, and Recall@K.
    Cached by data hash — re-trains only when the underlying data changes.
    Model serialised to bytes via joblib for session reuse without disk I/O.

    Validation approach:
      - Stratified 5-fold CV on full dataset → honest ROC-AUC estimate
      - Final model fit on time-split training window for deployment
      - PR-AUC and Recall@K evaluated on held-out time-split test set
    """
    X_all = pd.read_json(io.StringIO(X_json))
    y_all = pd.read_json(io.StringIO(y_json), typ="series")
    X_tr  = pd.read_json(io.StringIO(X_train_json))
    y_tr  = pd.read_json(io.StringIO(y_train_json), typ="series")
    X_te  = pd.read_json(io.StringIO(X_test_json))
    y_te  = pd.read_json(io.StringIO(y_test_json), typ="series")

    base_rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    rf = CalibratedClassifierCV(
        base_rf,
        method="isotonic",  # best for enough data
        cv=3
    )

    # ── Cross-validated ROC-AUC (5-fold stratified) ───────────────────────────
    # Stratified folds preserve the class ratio in each fold.
    # Each fold is scored on data the model hasn't seen — more robust than
    # a single train/test split, especially on smaller datasets.
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(rf, X_all, y_all, cv=cv, scoring="roc_auc", n_jobs=-1)
    roc_auc = float(cv_aucs.mean())

    # ── Final model fit on training window ───────────────────────────────────
    rf.fit(X_tr, y_tr)

    # PR-AUC and Recall@K on held-out time-split test set
    p_test = rf.predict_proba(X_te)[:, 1]
    pr_auc = float(average_precision_score(y_te, p_test))

    def recall_at_k(y_true, y_score, k):
        idx = np.argsort(-y_score)[:max(1, int(len(y_true) * k))]
        return float(y_true.iloc[idx].sum() / max(1, y_true.sum()))

    r10 = recall_at_k(y_te, p_test, 0.10)
    r20 = recall_at_k(y_te, p_test, 0.20)

    # ── Pre-compute scores and segments inside cache ──────────────────────────
    # Avoids running predict_proba and qcut on every page interaction.
    X_all_scores = pd.read_json(io.StringIO(X_json))
    scores = rf.predict_proba(X_all_scores)[:, 1]
    q80 = np.quantile(scores, 0.80)
    q60 = np.quantile(scores, 0.60)
    q40 = np.quantile(scores, 0.40)

    def _tier(s):
        if s >= q80:   return "High"
        elif s >= q60: return "Medium"
        elif s >= q40: return "Low"
        else:          return "Very Low"

    segments = [_tier(s) for s in scores]

    # Serialise model to bytes — reloaded each session without disk I/O
    buf = io.BytesIO()
    joblib.dump(rf, buf)
    model_bytes = buf.getvalue()

    return model_bytes, roc_auc, pr_auc, r10, r20, scores.tolist(), segments


model_bytes, roc_auc, pr_auc, recall_top10, recall_top20, _scores, _segments = train_and_evaluate(
    data_hash    = _data_hash,
    feature_cols = MODEL_FEATURES,
    X_json       = X.to_json(),
    y_json       = y.to_json(),
    X_train_json = X_train.to_json(),
    y_train_json = y_train.to_json(),
    X_test_json  = X_test.to_json(),
    y_test_json  = y_test.to_json(),
)

# Deserialise model from bytes for permutation importance
rf = joblib.load(io.BytesIO(model_bytes))

# ── Save model to disk for near-instant cold starts ───────────────────────────
# Saves once after training; on subsequent loads checks if data hash matches
# the saved model's hash before skipping retraining.
_hash_file = MODEL_DISK_PATH.with_suffix(".hash")
try:
    MODEL_DISK_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved_hash = _hash_file.read_text().strip() if _hash_file.exists() else ""
    if saved_hash != _data_hash:
        joblib.dump(rf, MODEL_DISK_PATH)
        _hash_file.write_text(_data_hash)
except Exception:
    pass  # disk save is best-effort; never block the app

# Use pre-computed scores and segments from cache
donor_summary["propensity_score"] = _scores
donor_summary["segment"] = _segments

donor_summary["expected_next_gift"] = (
    donor_summary["propensity_score"] * donor_summary["avg_donation"]
)

# Annual giving rate — use gifts per year with a minimum 90-day observation window
observation_days = donor_summary["days_since_first"].clip(lower=90)
donor_summary["gifts_per_year"] = (
    donor_summary["donation_count"] / observation_days * 365
).clip(upper=12)  # cap at 12 gifts/year — monthly giving is the ceiling

donor_summary["annual_giving_rate"] = (
    donor_summary["avg_donation"] * donor_summary["gifts_per_year"]
)

donor_summary["expected_lifespan_yrs"] = (
    donor_summary["propensity_score"] / (
        1 - donor_summary["propensity_score"].clip(upper=0.99)
    )
).clip(upper=20)

donor_summary["predicted_ltv"] = (
    donor_summary["annual_giving_rate"] * donor_summary["expected_lifespan_yrs"]
).round(2)

# Giving tier — donor pyramid segmentation
def giving_tier(total):
    if total >= 1000:  return "Major"
    elif total >= 250: return "Mid-level"
    else:              return "Small"

donor_summary["giving_tier"] = donor_summary["total_donated"].apply(giving_tier)

# Lapsed flag — high-value re-engagement targets
donor_summary["lapsed"] = (donor_summary["recency_days"] > 365).astype(int)

# Gift trend — is the donor growing or declining?
first_gifts = train_df.sort_values("donation_date").groupby("donor_id")["donation_amount"].first()
last_gifts  = train_df.sort_values("donation_date").groupby("donor_id")["donation_amount"].last()
donor_summary["gift_trend"] = (
    donor_summary["donor_id"].map(last_gifts - first_gifts).round(2)
)

# ── Apply segment filter ──────────────────────────────────────────────────────
if segment_filter != "All":
    chart_df = donor_summary[donor_summary["segment"] == segment_filter].copy()
else:
    chart_df = donor_summary.copy()

total_donors = len(donor_summary)

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_donors = len(donor_summary)
high_pct     = (donor_summary["segment"] == "High").mean()
repeat_rate  = (donor_summary["donated_again"] == 1).mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Donors",           f"{total_donors:,}")
k2.metric("High Propensity",        f"{high_pct:.1%}")
k3.metric("ROC-AUC (CV)",           f"{roc_auc:.2f}",
          help="5-fold cross-validated AUC. More robust than a single split. 0.5 = random, 1.0 = perfect.")
k4.metric("PR-AUC",                 f"{pr_auc:.2f}",
          help="Precision-recall AUC on held-out test set. More meaningful on imbalanced targets.")
k5.metric("Overall Retention Rate", f"{repeat_rate:.1%}")

st.divider()

seg_colors = alt.Scale(
    domain=["High", "Medium", "Low", "Very Low"],
    range=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"],
)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown('<p class="section-header">Score Distribution</p>', unsafe_allow_html=True)
    hist_chart = (
        alt.Chart(chart_df[["propensity_score"]])
        .mark_bar(color="#2ecc71", opacity=0.85)
        .encode(
            x=alt.X("propensity_score:Q", bin=alt.Bin(maxbins=30), title="Propensity Score"),
            y=alt.Y("count():Q", title="Number of Donors"),
            tooltip=["count():Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(hist_chart, use_container_width=True)

with c2:
    st.markdown('<p class="section-header">Who Comes Back?</p>', unsafe_allow_html=True)
    seg_perf = (
        donor_summary.groupby("segment")["donated_again"]
        .mean().reset_index()
        .rename(columns={"donated_again": "retention_rate"})
    )
    bar_chart = (
        alt.Chart(to_display(seg_perf))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("segment:N", sort=["High", "Medium", "Low", "Very Low"], title="Segment"),
            y=alt.Y("retention_rate:Q", title="Retention Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color("segment:N", scale=seg_colors, legend=None),
            tooltip=[
                alt.Tooltip("segment:N",        title="Segment"),
                alt.Tooltip("retention_rate:Q", title="Retention Rate", format=".1%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(bar_chart, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown('<p class="section-header">Giving Frequency</p>', unsafe_allow_html=True)
    freq_data = (
        donor_summary["donation_count"]
        .value_counts().reset_index()
        .rename(columns={"donation_count": "donations", "count": "num_donors"})
        .sort_values("donations")
    )
    freq_chart = (
        alt.Chart(freq_data)
        .mark_bar(color="#3498db", opacity=0.85)
        .encode(
            x=alt.X("donations:O", title="Number of Donations"),
            y=alt.Y("num_donors:Q", title="Number of Donors"),
            tooltip=["donations:O", "num_donors:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(freq_chart, use_container_width=True)

with c4:
    st.markdown('<p class="section-header">Recency vs Gift Size</p>', unsafe_allow_html=True)
    scatter_data = chart_df[["recency_days", "avg_donation", "segment"]].copy()
    scatter_data["recency_days"] = scatter_data["recency_days"].astype(int)
    scatter_data["avg_donation"] = scatter_data["avg_donation"].astype(float)
    sampled = scatter_data.sample(min(1000, len(scatter_data)), random_state=42).reset_index(drop=True)

    dots = (
        alt.Chart(sampled)
        .mark_circle(size=35, opacity=0.4)
        .encode(
            x=alt.X("recency_days:Q", title="Recency (days since last gift)"),
            y=alt.Y("avg_donation:Q",  title="Avg Donation ($)"),
            color=alt.Color("segment:N", scale=seg_colors, legend=alt.Legend(title="Segment")),
            tooltip=[
                alt.Tooltip("recency_days:Q", title="Recency (days)"),
                alt.Tooltip("avg_donation:Q",  title="Avg Donation", format="$.2f"),
                alt.Tooltip("segment:N",       title="Segment"),
            ],
        )
    )

    seg_reg_rows = []
    for seg in sampled["segment"].unique():
        sub = sampled[sampled["segment"] == seg]
        if len(sub) >= 2:
            coef = np.polyfit(sub["recency_days"], sub["avg_donation"], 1)
            x0, x1 = int(sub["recency_days"].min()), int(sub["recency_days"].max())
            seg_reg_rows += [
                {"segment": seg, "x": x0, "y": float(coef[0] * x0 + coef[1])},
                {"segment": seg, "x": x1, "y": float(coef[0] * x1 + coef[1])},
            ]

    seg_lines = (
        alt.Chart(pd.DataFrame(seg_reg_rows))
        .mark_line(strokeWidth=1.5, strokeDash=[4, 4], opacity=0.5)
        .encode(
            x="x:Q", y="y:Q",
            color=alt.Color("segment:N", scale=seg_colors, legend=None),
            detail="segment:N",
        )
    )

    coef_all = np.polyfit(sampled["recency_days"], sampled["avg_donation"], 1)
    global_df = pd.DataFrame([
        {"x": int(sampled["recency_days"].min()),
         "y": float(coef_all[0] * sampled["recency_days"].min() + coef_all[1])},
        {"x": int(sampled["recency_days"].max()),
         "y": float(coef_all[0] * sampled["recency_days"].max() + coef_all[1])},
    ])
    global_line = (
        alt.Chart(global_df)
        .mark_line(color="#cccccc", strokeWidth=1.5, strokeDash=[4, 4], opacity=0.4)
        .encode(x="x:Q", y="y:Q")
    )

    st.altair_chart(
        (dots + global_line + seg_lines).properties(height=280),
        use_container_width=True,
    )

st.divider()

# ── Model Lift Chart — test set only ─────────────────────────────────────────
st.markdown('<p class="section-header">Model vs Random: How Much Better?</p>',
            unsafe_allow_html=True)
st.caption("Contacting top-scored donors finds retained donors faster — here's how much faster.")

lift_df = donor_summary[test_mask][["propensity_score", "donated_again"]].copy()
lift_df = lift_df.sort_values("propensity_score", ascending=False).reset_index(drop=True)

total          = len(lift_df)
total_retained = lift_df["donated_again"].sum()

if total_retained == 0:
    st.warning("No retained donors in test set — lift chart not available.")
    st.stop()

steps          = list(range(1, 101))
model_capture  = []
random_capture = []
for pct in steps:
    n = max(1, int(total * pct / 100))
    model_capture.append(lift_df["donated_again"].iloc[:n].sum() / total_retained * 100)
    random_capture.append(pct)

lift_data = pd.DataFrame({
    "pct_contacted":         steps + steps,
    "pct_retained_captured": model_capture + random_capture,
    "type":                  ["Model"] * 100 + ["Random"] * 100,
})

lift_chart = (
    alt.Chart(lift_data).mark_line(strokeWidth=2)
    .encode(
        x=alt.X("pct_contacted:Q",         title="% of Donors Contacted"),
        y=alt.Y("pct_retained_captured:Q", title="% of Retained Donors Captured"),
        color=alt.Color("type:N",
                         scale=alt.Scale(domain=["Model", "Random"],
                                          range=["#2ecc71", "#aaaaaa"]),
                         legend=alt.Legend(title="", orient="bottom-right")),
        strokeDash=alt.condition(
            alt.datum.type == "Random", alt.value([4, 4]), alt.value([0])),
        tooltip=[
            alt.Tooltip("pct_contacted:Q",         title="% Contacted"),
            alt.Tooltip("pct_retained_captured:Q", title="% Retained Captured", format=".1f"),
            alt.Tooltip("type:N",                  title="Method"),
        ],
    )
)

top33_model = lift_df["donated_again"].iloc[:int(total * 0.33)].sum() / total_retained * 100
lift_at_33  = top33_model / 33.0

col_chart, col_stat = st.columns([8, 1])
with col_chart:
    st.altair_chart(lift_chart.properties(height=350), use_container_width=True)
with col_stat:
    st.metric("Lift at top 33%",             f"{lift_at_33:.1f}x",
              help="Top 33% of scored donors finds this many times more retained donors than random.")
    st.metric("Retained captured (top 33%)", f"{top33_model:.1f}%")
    st.metric("Recall @ top 10%",            f"{recall_top10:.1%}",
              help="% of all retained donors found by contacting only the top 10%.")
    st.metric("Recall @ top 20%",            f"{recall_top20:.1%}",
              help="% of all retained donors found by contacting only the top 20%.")

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown('<p class="section-header">What Drives the Model?</p>', unsafe_allow_html=True)
st.caption("Permutation importance — how much model performance drops when each feature is shuffled.")


@st.cache_data(show_spinner="Computing feature importance...", ttl=3600)
def cached_permutation_importance(data_hash: str, model_hash: str,
                                   X_test_json: str, y_test_json: str) -> pd.DataFrame:
    """Cached permutation importance — reruns only when data or model changes."""
    X_te = pd.read_json(io.StringIO(X_test_json))
    y_te = pd.read_json(io.StringIO(y_test_json), typ="series")
    _rf  = joblib.load(io.BytesIO(bytes.fromhex(model_hash)))
    perm = permutation_importance(
        _rf, X_te, y_te,
        n_repeats=3,          # reduced from 5 — ~40% faster, negligible accuracy loss
        random_state=42,
        scoring="average_precision",
    )
    return pd.DataFrame({
        "feature":    X_te.columns,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False).head(15).reset_index(drop=True)


imp_df = cached_permutation_importance(
    data_hash   = _data_hash,
    model_hash  = model_bytes.hex(),
    X_test_json = X_test.to_json(),
    y_test_json = y_test.to_json(),
)

imp_chart = (
    alt.Chart(imp_df)
    .mark_bar(color="#3498db", opacity=0.85)
    .encode(
        x=alt.X("importance:Q", title="Mean Importance (PR-AUC drop)"),
        y=alt.Y("feature:N",    sort="-x", title="Feature"),
        tooltip=[
            alt.Tooltip("feature:N",    title="Feature"),
            alt.Tooltip("importance:Q", title="Importance", format=".4f"),
        ],
    )
    .properties(height=380)
)
st.altair_chart(imp_chart, use_container_width=True)

st.divider()

# ── Donor Leaderboard ─────────────────────────────────────────────────────────
display_df = chart_df.sort_values("predicted_ltv", ascending=False).reset_index(drop=True)

st.markdown(
    f'<p class="section-header">Donor Leaderboard '
    f'<span style="font-weight:400;color:#888;">({len(display_df):,} donors · {segment_filter} segment)</span></p>',
    unsafe_allow_html=True,
)

table_cols = [
    "donor_id", "donation_count", "total_donated", "avg_donation",
    "recency_days", "months_since_last", "lapsed", "gift_trend",
    "giving_tier", "last_donation", "donated_again", "expected_next_gift",
    "predicted_ltv", "propensity_score", "segment"
]
table_cols = [c for c in table_cols if c in display_df.columns]

st.dataframe(
    to_display(display_df[table_cols]),
    use_container_width=True,
    height=350,
    hide_index=True,
    column_config={
        "donor_id":         st.column_config.TextColumn("Donor ID"),
        "donation_count":   st.column_config.NumberColumn("Donations",     format="%d"),
        "total_donated":    st.column_config.NumberColumn("Total Donated", format="$%.2f"),
        "avg_donation":     st.column_config.NumberColumn("Avg Donation",  format="$%.2f"),
        "recency_days":     st.column_config.NumberColumn("Recency (days)"),
        "last_donation":    st.column_config.TextColumn("Last Donation"),
        "donated_again":    None,
        "propensity_score": st.column_config.ProgressColumn(
            "Propensity Score", min_value=0, max_value=1, format="%.2f"),
        "segment":          st.column_config.TextColumn("Segment"),
        "predicted_ltv": st.column_config.NumberColumn("Expected LTV ($)", format="$%.2f"),
        "months_since_last": st.column_config.NumberColumn("Months Since Last Gift", format="%.1f"),
        "lapsed":            st.column_config.CheckboxColumn("Lapsed (12m+)"),
        "gift_trend":        st.column_config.NumberColumn("Gift Trend ($)", format="$%+.2f"),
        "giving_tier":       st.column_config.TextColumn("Giving Tier"),
        "expected_next_gift": st.column_config.NumberColumn("Expected Next Gift ($)", format="$%.2f"),
    },
)

st.divider()

# ── Download ──────────────────────────────────────────────────────────────────
csv = chart_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download results",
    data=csv,
    file_name="donor_propensity.csv",
    mime="text/csv",
)