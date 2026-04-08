"""
1_Donor_Propensity.py

Donor propensity scoring dashboard.
Model logic:
  - RandomForestClassifier (handles nonlinear interactions, robust to outliers)
  - Time-based train/test split (mirrors real deployment: train on past, score present)
  - Full feature set: RFM + engagement + donor profile + giving behaviour flags
  - 4-tier segmentation: High / Medium / Low / Very Low
  - ROC-AUC + PR-AUC + Recall@K metrics
  - Permutation feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
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


# ── Header ────────────────────────────────────────────────────────────────────
st.title("💚 Donor Propensity Scoring")
st.caption("Score, segment, and prioritize your donor base for maximum retention impact.")
st.divider()

# ── How to use ────────────────────────────────────────────────────────────────
with st.expander("📖 How to use this tool", expanded=False):
    st.markdown("""
    **This dashboard scores your donors by likelihood to give again and segments them into four tiers.**
    Use the sidebar filter to focus on a specific segment at any time.

    ---

    **🟢 High — Priority outreach**
    Strongest signals for repeat giving. Contact these first.

    **🟡 Medium — Nurture**
    Moderate engagement. A lighter touch can move them toward High.

    **🟠 Low — Monitor**
    Weak retention signals. Periodic low-cost engagement only.

    **🔴 Very Low — Deprioritize**
    Minimal signals. Consider re-engagement campaigns or exclude from outreach.

    ---

    **Reading the charts**
    - **Score Distribution** — scores near 1.0 = high likelihood to give again
    - **Who Comes Back?** — retention rate per tier confirms model is working
    - **Giving Frequency** — most donors give once; repeat donors are rare and valuable
    - **Recency vs Gift Size** — lapsed high-value donors are strong re-engagement targets
    - **Model vs Random** — green line above diagonal = model beats random selection
    - **Feature Importance** — which signals drive the model most

    ---

    **Suggested next steps**
    1. Export the **High tier** list for immediate outreach
    2. For **Medium donors** with high total giving, consider a personalized ask
    3. Re-run the model monthly as new donation data comes in
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
# Mirrors real deployment — model trained on historical data, scored on current
cutoff_date = df["donation_date"].quantile(0.8)
train_df    = df[df["donation_date"] <= cutoff_date]
future_df   = df[df["donation_date"] >  cutoff_date]

today = df["donation_date"].max()

# ── Feature engineering per donor ─────────────────────────────────────────────
# Build features from training window only to prevent leakage

def build_donor_features(transactions: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build donor features from transaction history.
    All features computed from the training window only.

    RFM core:
      recency_days, donation_count, total_donated, avg_donation, amount_log

    Giving behaviour flags:
      donor_last_3yrs         — gave in last 3 years
      gift_100_last_3yrs      — any gift >= $100 in last 3 years 
      gift_100_lifetime       — any gift >= $100 ever
      has_recurring           — ever made a recurring gift
      never_donated_flag      — no prior donation (always 0 here by construction)
      first_gift_missing_flag — missing first gift date

    Engagement / profile (from richer dataset):
      newsletter_opt_in       — opted into newsletter
      referral_channel_*      — how they first found the org (one-hot)
      age_group_*             — donor age group (one-hot)

    Time-based:
      days_since_first_gift   — tenure as donor
      giving_frequency_ratio  — gifts per day active
      months_since_last_gift  — recency in months

    Pipeline:
      stage_score             — proxy for cultivation stage (donation_count-based)
    """
    g = transactions.groupby("donor_id")

    summary = g.agg(
        donation_count    =("donation_id",     "count"),
        total_donated     =("donation_amount",  "sum"),
        avg_donation      =("donation_amount",  "mean"),
        max_donation      =("donation_amount",  "max"),
        last_donation     =("donation_date",    "max"),
        first_donation    =("donation_date",    "min"),
    ).reset_index()

    summary["recency_days"]       = (ref_date - summary["last_donation"]).dt.days
    summary["months_since_last"]  = summary["recency_days"] / 30.44
    summary["days_since_first"]   = (ref_date - summary["first_donation"]).dt.days
    summary["amount_log"]         = np.log1p(summary["total_donated"])
    summary["avg_donation_log"]   = np.log1p(summary["avg_donation"])

    # Giving frequency ratio: gifts per day active (avoid div by zero)
    active_days = summary["days_since_first"].replace(0, 1)
    summary["giving_frequency_ratio"] = summary["donation_count"] / active_days

    # Stage score proxy: log of donation count (more donations = further cultivated)
    summary["stage_score"] = np.log1p(summary["donation_count"])

    # Missing flags
    summary["first_gift_missing_flag"] = summary["first_donation"].isna().astype(int)
    summary["never_donated_flag"]      = 0  # all donors here have donated

    # 3-year lookback window
    three_yrs_ago = ref_date - pd.DateOffset(years=3)
    recent = transactions[transactions["donation_date"] >= three_yrs_ago]
    recent_agg = recent.groupby("donor_id").agg(
        donor_last_3yrs    =("donation_id",    "count"),
        gift_100_last_3yrs =("donation_amount", lambda x: int((x >= 100).any())),
    ).reset_index()
    recent_agg["donor_last_3yrs"] = (recent_agg["donor_last_3yrs"] > 0).astype(int)

    summary = summary.merge(recent_agg, on="donor_id", how="left")
    summary["donor_last_3yrs"]    = summary["donor_last_3yrs"].fillna(0).astype(int)
    summary["gift_100_last_3yrs"] = summary["gift_100_last_3yrs"].fillna(0).astype(int)

    # Lifetime flags
    lifetime = transactions.groupby("donor_id").agg(
        gift_100_lifetime =("donation_amount",  lambda x: int((x >= 100).any())),
        has_recurring     =("donation_type",    lambda x: int((x == "Recurring").any())),
    ).reset_index()
    summary = summary.merge(lifetime, on="donor_id", how="left")

    # Newsletter opt-in (take most recent value per donor)
    if "newsletter_opt_in" in transactions.columns:
        nl = transactions.sort_values("donation_date").groupby("donor_id")["newsletter_opt_in"].last().reset_index()
        nl["newsletter_opt_in"] = nl["newsletter_opt_in"].astype(int)
        summary = summary.merge(nl, on="donor_id", how="left")
        summary["newsletter_opt_in"] = summary["newsletter_opt_in"].fillna(0).astype(int)

    # Referral channel (most common per donor → one-hot)
    if "referral_channel" in transactions.columns:
        ref = transactions.groupby("donor_id")["referral_channel"].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index()
        ref_dummies = pd.get_dummies(ref["referral_channel"], prefix="ref").astype(int)
        ref = pd.concat([ref[["donor_id"]], ref_dummies], axis=1)
        summary = summary.merge(ref, on="donor_id", how="left")
        for c in [col for col in summary.columns if col.startswith("ref_")]:
            summary[c] = summary[c].fillna(0).astype(int)

    # Age group (most common per donor → one-hot)
    if "age_group" in transactions.columns:
        age = transactions.groupby("donor_id")["age_group"].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index()
        age_dummies = pd.get_dummies(age["age_group"], prefix="age").astype(int)
        age = pd.concat([age[["donor_id"]], age_dummies], axis=1)
        summary = summary.merge(age, on="donor_id", how="left")
        for c in [col for col in summary.columns if col.startswith("age_")]:
            summary[c] = summary[c].fillna(0).astype(int)

    return summary


donor_summary = build_donor_features(train_df, cutoff_date)

# Target: did the donor give again in the future window?
future_donors = future_df["donor_id"].unique()
donor_summary["donated_again"] = donor_summary["donor_id"].isin(future_donors).astype(int)

# ── Model features ────────────────────────────────────────────────────────────
BASE_FEATURES = [
    # Core RFM — genuinely predictive
    "recency_days",
    "donation_count",
    "amount_log",
    "avg_donation_log",
    "months_since_last",
    "days_since_first",
    "giving_frequency_ratio",
    # Lifetime flags only — not windowed
    "gift_100_lifetime",
    "first_gift_missing_flag",
    "never_donated_flag",
    # Engagement
    "newsletter_opt_in",
    # Pipeline proxy
    "stage_score",
]

# Add one-hot encoded columns that were generated
EXTRA_FEATURES = [c for c in donor_summary.columns
                  if c.startswith("ref_") or c.startswith("age_")]

MODEL_FEATURES = [f for f in BASE_FEATURES + EXTRA_FEATURES if f in donor_summary.columns]

X = donor_summary[MODEL_FEATURES].fillna(0)
y = donor_summary["donated_again"]

# ── Time-based train/test split ───────────────────────────────────────────────
# Use 80th percentile of last_donation as the cutoff within the donor summary
split_cutoff = donor_summary["last_donation"].quantile(0.8)
train_mask   = donor_summary["last_donation"] <= split_cutoff
test_mask    = donor_summary["last_donation"] >  split_cutoff

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# Fallback to full data if test set is too small
if len(X_test) < 50 or y_test.nunique() < 2:
    X_train, X_test = X, X
    y_train, y_test = y, y

# ── Train RandomForest ────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# ── Metrics ───────────────────────────────────────────────────────────────────
p_test  = rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, p_test)
pr_auc  = average_precision_score(y_test, p_test)

def recall_at_k(y_true, y_score, k=0.10):
    n   = len(y_true)
    idx = np.argsort(-y_score)[:max(1, int(n * k))]
    return y_true.iloc[idx].sum() / max(1, y_true.sum())

recall_top10 = recall_at_k(y_test, p_test, k=0.10)
recall_top20 = recall_at_k(y_test, p_test, k=0.20)

# ── Score all donors ──────────────────────────────────────────────────────────
donor_summary["propensity_score"] = rf.predict_proba(X)[:, 1]

# 4-tier segmentation (percentile-based)
q80 = donor_summary["propensity_score"].quantile(0.80)
q60 = donor_summary["propensity_score"].quantile(0.60)
q40 = donor_summary["propensity_score"].quantile(0.40)

def assign_tier(s):
    if s >= q80:   return "High"
    elif s >= q60: return "Medium"
    elif s >= q40: return "Low"
    else:          return "Very Low"

donor_summary["segment"] = donor_summary["propensity_score"].apply(assign_tier)

# ── Apply segment filter ──────────────────────────────────────────────────────
if segment_filter != "All":
    chart_df = donor_summary[donor_summary["segment"] == segment_filter].copy()
else:
    chart_df = donor_summary.copy()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_donors = len(donor_summary)
high_pct     = (donor_summary["segment"] == "High").mean()
repeat_rate  = (donor_summary["donated_again"] == 1).mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Donors",          f"{total_donors:,}")
k2.metric("High Propensity",       f"{high_pct:.1%}")
k3.metric("ROC-AUC",               f"{roc_auc:.2f}", help="Overall ranking quality. 0.5 = random, 1.0 = perfect.")
k4.metric("PR-AUC",                f"{pr_auc:.2f}",  help="Precision-recall quality. More meaningful on imbalanced targets.")
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
                alt.Tooltip("segment:N",      title="Segment"),
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
            x0   = int(sub["recency_days"].min())
            x1   = int(sub["recency_days"].max())
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

# ── Model Lift Chart ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Model vs Random: How Much Better?</p>',
            unsafe_allow_html=True)
st.caption("Contacting top-scored donors finds retained donors faster — here's how much faster.")

# Compute lift on test set only — avoids in-sample inflation
lift_df = donor_summary[test_mask][["propensity_score", "donated_again"]].copy()
lift_df = lift_df.sort_values("propensity_score", ascending=False).reset_index(drop=True)

total          = len(lift_df)
total_retained = lift_df["donated_again"].sum()

steps         = list(range(1, 101))
model_capture = []
random_capture = []
for pct in steps:
    n = max(1, int(total * pct / 100))
    model_capture.append(lift_df["donated_again"].iloc[:n].sum() / total_retained * 100)
    random_capture.append(pct)

lift_data = pd.DataFrame({
    "pct_contacted":       steps + steps,
    "pct_retained_captured": model_capture + random_capture,
    "type":                ["Model"] * 100 + ["Random"] * 100,
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
    st.metric("Lift at top 33%",          f"{lift_at_33:.1f}x",
              help="Top 33% of scored donors finds this many times more retained donors than random.")
    st.metric("Retained captured (top 33%)", f"{top33_model:.1f}%")
    st.metric("Recall @ top 10%",         f"{recall_top10:.1%}",
              help="% of all retained donors found by contacting only the top 10%.")
    st.metric("Recall @ top 20%",         f"{recall_top20:.1%}",
              help="% of all retained donors found by contacting only the top 20%.")

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown('<p class="section-header">What Drives the Model?</p>', unsafe_allow_html=True)
st.caption("Permutation importance — how much model performance drops when each feature is shuffled.")

with st.spinner("Computing feature importance..."):
    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=5,
        random_state=42,
        scoring="average_precision",
    )

imp_df = (
    pd.DataFrame({
        "feature":    X_test.columns,
        "importance": perm.importances_mean,
    })
    .sort_values("importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
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
display_df = chart_df.sort_values("propensity_score", ascending=False).reset_index(drop=True)

st.markdown(
    f'<p class="section-header">Donor Leaderboard '
    f'<span style="font-weight:400;color:#888;">({len(display_df):,} donors · {segment_filter} segment)</span></p>',
    unsafe_allow_html=True,
)

table_cols = ["donor_id", "donation_count", "total_donated", "avg_donation",
              "recency_days", "last_donation", "donated_again", "propensity_score", "segment"]
table_cols = [c for c in table_cols if c in display_df.columns]

st.dataframe(
    to_display(display_df[table_cols]),
    use_container_width=True,
    height=350,
    hide_index=True,
    column_config={
        "donor_id":        st.column_config.TextColumn("Donor ID"),
        "donation_count":  st.column_config.NumberColumn("Donations", format="%d"),
        "total_donated":   st.column_config.NumberColumn("Total Donated", format="$%.2f"),
        "avg_donation":    st.column_config.NumberColumn("Avg Donation",  format="$%.2f"),
        "recency_days":    st.column_config.NumberColumn("Recency (days)"),
        "last_donation":   st.column_config.TextColumn("Last Donation"),
        "donated_again":   None,
        "propensity_score": st.column_config.ProgressColumn(
            "Propensity Score", min_value=0, max_value=1, format="%.2f"),
        "segment":         st.column_config.TextColumn("Segment"),
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