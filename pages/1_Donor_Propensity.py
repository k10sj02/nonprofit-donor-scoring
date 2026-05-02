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
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Donor Propensity Scoring",
    page_icon="💚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #4a4a4a; margin-bottom: 0.25rem; }
    [data-testid="stHorizontalBlock"] { align-items: stretch; }
    .vega-embed { width: 100% !important; }
    .vega-embed canvas { width: 100% !important; }
    </style>
""",
    unsafe_allow_html=True,
)


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
st.caption(
    "Score, segment, and prioritize your donor base for maximum retention impact."
)
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
# Load pre-trained model artifact
# Model is trained locally via train_donor_model.py and committed to the repo.
# No training happens at runtime — mirrors the pattern in train_models.py.
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "outputs" / "donor_model.joblib"


@st.cache_resource(show_spinner="Loading model…")
def load_artifact(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


artifact = load_artifact(str(MODEL_ARTIFACT_PATH))

if artifact is None:
    st.error(
        "Model artifact not found. Run `uv run python train_donor_model.py` locally "
        "and commit `outputs/donor_model.joblib` to the repo."
    )
    st.stop()

# Unpack artifact
donor_summary = artifact["donor_summary"].copy()
_scores = artifact["scores"]
_segments = artifact["segments"]
roc_auc = artifact["roc_auc"]
pr_auc = artifact["pr_auc"]
recall_top10 = artifact["recall_top10"]
recall_top20 = artifact["recall_top20"]
imp_df = artifact["imp_df"]
test_mask = pd.Series(artifact["test_mask"], index=donor_summary.index)
rf = artifact["model"]

# Data prep for leaderboard display only (no training)
df["donation_date"] = pd.to_datetime(df["donation_date"])
cutoff_date = df["donation_date"].quantile(0.8)
train_df = df[df["donation_date"] <= cutoff_date]


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
).clip(
    upper=12
)  # cap at 12 gifts/year — monthly giving is the ceiling

donor_summary["annual_giving_rate"] = (
    donor_summary["avg_donation"] * donor_summary["gifts_per_year"]
)

donor_summary["expected_lifespan_yrs"] = (
    donor_summary["propensity_score"]
    / (1 - donor_summary["propensity_score"].clip(upper=0.99))
).clip(upper=20)

donor_summary["predicted_ltv"] = (
    donor_summary["annual_giving_rate"] * donor_summary["expected_lifespan_yrs"]
).round(2)


# Giving tier — donor pyramid segmentation
def giving_tier(total):
    if total >= 1000:
        return "Major"
    elif total >= 250:
        return "Mid-level"
    else:
        return "Small"


donor_summary["giving_tier"] = donor_summary["total_donated"].apply(giving_tier)

# Lapsed flag — high-value re-engagement targets
donor_summary["lapsed"] = (donor_summary["recency_days"] > 365).astype(int)

# Gift trend — is the donor growing or declining?
first_gifts = (
    train_df.sort_values("donation_date").groupby("donor_id")["donation_amount"].first()
)
last_gifts = (
    train_df.sort_values("donation_date").groupby("donor_id")["donation_amount"].last()
)
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
high_pct = (donor_summary["segment"] == "High").mean()
repeat_rate = (donor_summary["donated_again"] == 1).mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Donors", f"{total_donors:,}")
k2.metric("High Propensity", f"{high_pct:.1%}")
k3.metric(
    "ROC-AUC (CV)",
    f"{roc_auc:.2f}",
    help="5-fold cross-validated AUC. More robust than a single split. 0.5 = random, 1.0 = perfect.",
)
k4.metric(
    "PR-AUC",
    f"{pr_auc:.2f}",
    help="Precision-recall AUC on held-out test set. More meaningful on imbalanced targets.",
)
k5.metric("Overall Retention Rate", f"{repeat_rate:.1%}")

st.divider()

seg_colors = alt.Scale(
    domain=["High", "Medium", "Low", "Very Low"],
    range=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"],
)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown(
        '<p class="section-header">Score Distribution</p>', unsafe_allow_html=True
    )
    hist_chart = (
        alt.Chart(chart_df[["propensity_score"]])
        .mark_bar(color="#2ecc71", opacity=0.85)
        .encode(
            x=alt.X(
                "propensity_score:Q", bin=alt.Bin(maxbins=30), title="Propensity Score"
            ),
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
        .mean()
        .reset_index()
        .rename(columns={"donated_again": "retention_rate"})
    )
    bar_chart = (
        alt.Chart(to_display(seg_perf))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "segment:N", sort=["High", "Medium", "Low", "Very Low"], title="Segment"
            ),
            y=alt.Y(
                "retention_rate:Q", title="Retention Rate", axis=alt.Axis(format=".0%")
            ),
            color=alt.Color("segment:N", scale=seg_colors, legend=None),
            tooltip=[
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("retention_rate:Q", title="Retention Rate", format=".1%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(bar_chart, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown(
        '<p class="section-header">Giving Frequency</p>', unsafe_allow_html=True
    )
    freq_data = (
        donor_summary["donation_count"]
        .value_counts()
        .reset_index()
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
    st.markdown(
        '<p class="section-header">Recency vs Gift Size</p>', unsafe_allow_html=True
    )
    scatter_data = chart_df[["recency_days", "avg_donation", "segment"]].copy()
    scatter_data["recency_days"] = scatter_data["recency_days"].astype(int)
    scatter_data["avg_donation"] = scatter_data["avg_donation"].astype(float)
    sampled = scatter_data.sample(
        min(1000, len(scatter_data)), random_state=42
    ).reset_index(drop=True)

    dots = (
        alt.Chart(sampled)
        .mark_circle(size=35, opacity=0.4)
        .encode(
            x=alt.X("recency_days:Q", title="Recency (days since last gift)"),
            y=alt.Y("avg_donation:Q", title="Avg Donation ($)"),
            color=alt.Color(
                "segment:N", scale=seg_colors, legend=alt.Legend(title="Segment")
            ),
            tooltip=[
                alt.Tooltip("recency_days:Q", title="Recency (days)"),
                alt.Tooltip("avg_donation:Q", title="Avg Donation", format="$.2f"),
                alt.Tooltip("segment:N", title="Segment"),
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
            x="x:Q",
            y="y:Q",
            color=alt.Color("segment:N", scale=seg_colors, legend=None),
            detail="segment:N",
        )
    )

    coef_all = np.polyfit(sampled["recency_days"], sampled["avg_donation"], 1)
    global_df = pd.DataFrame(
        [
            {
                "x": int(sampled["recency_days"].min()),
                "y": float(coef_all[0] * sampled["recency_days"].min() + coef_all[1]),
            },
            {
                "x": int(sampled["recency_days"].max()),
                "y": float(coef_all[0] * sampled["recency_days"].max() + coef_all[1]),
            },
        ]
    )
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
st.markdown(
    '<p class="section-header">Model vs Random: How Much Better?</p>',
    unsafe_allow_html=True,
)
st.caption(
    "Contacting top-scored donors finds retained donors faster — here's how much faster."
)

lift_df = donor_summary[test_mask][["propensity_score", "donated_again"]].copy()
lift_df = lift_df.sort_values("propensity_score", ascending=False).reset_index(
    drop=True
)

total = len(lift_df)
total_retained = lift_df["donated_again"].sum()

if total_retained == 0:
    st.warning("No retained donors in test set — lift chart not available.")
    st.stop()

steps = list(range(1, 101))
model_capture = []
random_capture = []
for pct in steps:
    n = max(1, int(total * pct / 100))
    model_capture.append(lift_df["donated_again"].iloc[:n].sum() / total_retained * 100)
    random_capture.append(pct)

lift_data = pd.DataFrame(
    {
        "pct_contacted": steps + steps,
        "pct_retained_captured": model_capture + random_capture,
        "type": ["Model"] * 100 + ["Random"] * 100,
    }
)

lift_chart = (
    alt.Chart(lift_data)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("pct_contacted:Q", title="% of Donors Contacted"),
        y=alt.Y("pct_retained_captured:Q", title="% of Retained Donors Captured"),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["Model", "Random"], range=["#2ecc71", "#aaaaaa"]),
            legend=alt.Legend(title="", orient="bottom-right"),
        ),
        strokeDash=alt.condition(
            alt.datum.type == "Random", alt.value([4, 4]), alt.value([0])
        ),
        tooltip=[
            alt.Tooltip("pct_contacted:Q", title="% Contacted"),
            alt.Tooltip(
                "pct_retained_captured:Q", title="% Retained Captured", format=".1f"
            ),
            alt.Tooltip("type:N", title="Method"),
        ],
    )
)

top33_model = (
    lift_df["donated_again"].iloc[: int(total * 0.33)].sum() / total_retained * 100
)
lift_at_33 = top33_model / 33.0

col_chart, col_stat = st.columns([8, 1])
with col_chart:
    st.altair_chart(lift_chart.properties(height=350), use_container_width=True)
with col_stat:
    st.metric(
        "Lift at top 33%",
        f"{lift_at_33:.1f}x",
        help="Top 33% of scored donors finds this many times more retained donors than random.",
    )
    st.metric("Retained captured (top 33%)", f"{top33_model:.1f}%")
    st.metric(
        "Recall @ top 10%",
        f"{recall_top10:.1%}",
        help="% of all retained donors found by contacting only the top 10%.",
    )
    st.metric(
        "Recall @ top 20%",
        f"{recall_top20:.1%}",
        help="% of all retained donors found by contacting only the top 20%.",
    )

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">What Drives the Model?</p>', unsafe_allow_html=True
)
st.caption(
    "Permutation importance — how much model performance drops when each feature is shuffled."
)


# imp_df loaded from pre-trained artifact


imp_chart = (
    alt.Chart(imp_df)
    .mark_bar(color="#3498db", opacity=0.85)
    .encode(
        x=alt.X("importance:Q", title="Mean Importance (PR-AUC drop)"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("importance:Q", title="Importance", format=".4f"),
        ],
    )
    .properties(height=380)
)
st.altair_chart(imp_chart, use_container_width=True)

st.divider()

# ── Donor Leaderboard ─────────────────────────────────────────────────────────
display_df = chart_df.sort_values("predicted_ltv", ascending=False).reset_index(
    drop=True
)

st.markdown(
    f'<p class="section-header">Donor Leaderboard '
    f'<span style="font-weight:400;color:#888;">({len(display_df):,} donors · {segment_filter} segment)</span></p>',
    unsafe_allow_html=True,
)

table_cols = [
    "donor_id",
    "donation_count",
    "total_donated",
    "avg_donation",
    "recency_days",
    "months_since_last",
    "lapsed",
    "gift_trend",
    "giving_tier",
    "last_donation",
    "donated_again",
    "expected_next_gift",
    "predicted_ltv",
    "propensity_score",
    "segment",
]
table_cols = [c for c in table_cols if c in display_df.columns]

st.dataframe(
    to_display(display_df[table_cols]),
    use_container_width=True,
    height=350,
    hide_index=True,
    column_config={
        "donor_id": st.column_config.TextColumn("Donor ID"),
        "donation_count": st.column_config.NumberColumn("Donations", format="%d"),
        "total_donated": st.column_config.NumberColumn("Total Donated", format="$%.2f"),
        "avg_donation": st.column_config.NumberColumn("Avg Donation", format="$%.2f"),
        "recency_days": st.column_config.NumberColumn("Recency (days)"),
        "last_donation": st.column_config.TextColumn("Last Donation"),
        "donated_again": None,
        "propensity_score": st.column_config.ProgressColumn(
            "Propensity Score", min_value=0, max_value=1, format="%.2f"
        ),
        "segment": st.column_config.TextColumn("Segment"),
        "predicted_ltv": st.column_config.NumberColumn(
            "Expected LTV ($)", format="$%.2f"
        ),
        "months_since_last": st.column_config.NumberColumn(
            "Months Since Last Gift", format="%.1f"
        ),
        "lapsed": st.column_config.CheckboxColumn("Lapsed (12m+)"),
        "gift_trend": st.column_config.NumberColumn("Gift Trend ($)", format="$%+.2f"),
        "giving_tier": st.column_config.TextColumn("Giving Tier"),
        "expected_next_gift": st.column_config.NumberColumn(
            "Expected Next Gift ($)", format="$%.2f"
        ),
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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
---
<div style="
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    color: #888;
    padding-top: 0.5rem;
">
    <div>Built by Stann-Omar Jones</div>
    <div>Donor Propensity Dashboard · v1.0</div>
</div>
""",
    unsafe_allow_html=True,
)
