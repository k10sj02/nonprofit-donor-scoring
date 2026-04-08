import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Opportunity Scoring",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4a4a4a;
        margin-bottom: 0.25rem;
    }
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


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Opportunity Scoring")
st.caption("Score and prioritize open pipeline opportunities by likelihood to close.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload opportunities CSV", type="csv")

    PROJECT_ROOT = Path(__file__).parent.parent
    default_path = PROJECT_ROOT / "data" / "synthetic" / "synthetic_opportunities.csv"

    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Using uploaded dataset")
    elif default_path.exists():
        df = pd.read_csv(default_path)
        st.info("Using default synthetic dataset")
    else:
        st.warning(
            "No dataset found. Run `generate_opportunities.py` first to generate "
            "`data/synthetic/synthetic_opportunities.csv`."
        )

    st.divider()
    st.markdown("**Filter by gift type**")
    gift_filter = st.radio(
        label="Gift Type",
        options=["All", "Annual Giving", "Major Gifts", "Planned Giving"],
        label_visibility="collapsed",
    )

if df is None:
    st.info("👈 Upload a dataset or run the generator script to get started.")
    st.stop()

# ── How to use ────────────────────────────────────────────────────────────────
with st.expander("📖 How to use this tool", expanded=False):
    st.markdown(
        """
    **This dashboard scores open pipeline opportunities by their likelihood to close successfully.**
    Use the sidebar filter to focus on a specific gift type.

    ---

    **Reading the pipeline stages**
    - **Discovery / Engagement** — early relationship building, low close probability
    - **Evaluation / Proposal** — active consideration, moderate probability
    - **Negotiation / Committed** — high probability, prioritize immediate action
    - **Lost / Rejected** — closed unsuccessfully, excluded from open pipeline scoring

    ---

    **Reading the charts**
    - **Pipeline by Stage** — deal count and total value at each stage
    - **Win Rate by Stage** — confirms model logic; later stages should win more
    - **Deal Size Distribution** — spread of opportunity amounts across gift types
    - **Close Probability vs Deal Size** — larger deals aren't always easier to close
    - **Model vs Random** — how much better the model is at finding winners vs picking randomly

    ---

    **Suggested next steps**
    1. Export the **Top Opportunities** list and share with your major gifts team
    2. Focus on high-score opportunities in **Negotiation** or **Proposal** stage
    3. For low-score opportunities with large deal amounts, investigate blockers
    4. Re-score monthly as pipeline stages and deal details are updated
    """
    )

# ── Data prep ─────────────────────────────────────────────────────────────────
df["stage_entry_date"] = pd.to_datetime(df["stage_entry_date"], errors="coerce")
df = df.rename(
    columns={
        "current_stage": "stage",
        "current_stage_probability_pct": "stage_probability_pct",
    }
)
df["is_closed"] = df["is_closed"].astype(bool)
df["is_successful"] = df["is_successful"].astype(bool)

# Stage ordering
stage_order = [
    "Discovery",
    "Engagement",
    "Evaluation",
    "Proposal",
    "Negotiation",
    "Committed",
    "Lost",
    "Rejected",
]
active_stages = ["Discovery", "Engagement", "Evaluation", "Proposal", "Negotiation"]

# ── ML Model — train on closed opps, score open ones ─────────────────────────
closed = df[df["is_closed"]].copy()
open_opps = df[~df["is_closed"]].copy()

# Encode categoricals
cat_cols = ["stage", "gift_type", "sector"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    closed[col + "_enc"] = le.fit_transform(closed[col].astype(str))
    encoders[col] = le

feature_cols = [
    "stage_enc",
    "gift_type_enc",
    "sector_enc",
    "stage_probability_pct",
    "deal_amount",
]

X_train = closed[
    [
        c
        for c in [f + "_enc" for f in cat_cols]
        + ["stage_probability_pct", "deal_amount"]
    ]
]
y_train = closed["is_successful"].astype(int)

from sklearn.model_selection import cross_val_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Cross-validated AUC to avoid inflated in-sample score
auc = cross_val_score(
    LogisticRegression(max_iter=1000), X_train, y_train,
    cv=5, scoring="roc_auc"
).mean()

# Score open opportunities
for col in cat_cols:
    le = encoders[col]
    known = set(le.classes_)
    open_opps[col + "_enc"] = (
        open_opps[col]
        .astype(str)
        .apply(lambda x: le.transform([x])[0] if x in known else 0)
    )

X_open = open_opps[
    [c + "_enc" for c in cat_cols] + ["stage_probability_pct", "deal_amount"]
]
open_opps["close_probability"] = model.predict_proba(X_open)[:, 1]

open_opps["priority"] = pd.qcut(
    open_opps["close_probability"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop",
).astype(str)

# ── Apply gift type filter ────────────────────────────────────────────────────
if gift_filter != "All":
    chart_open = open_opps[open_opps["gift_type"] == gift_filter].copy()
    chart_all = df[df["gift_type"] == gift_filter].copy()
else:
    chart_open = open_opps.copy()
    chart_all = df.copy()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_open = len(open_opps)
total_pipeline = open_opps["deal_amount"].sum()
win_rate = closed["is_successful"].mean()
high_priority_pct = (open_opps["priority"] == "High").mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Open Opportunities", f"{total_open:,}")
k2.metric("Total Pipeline Value", f"${total_pipeline:,.0f}")
k3.metric("Historical Win Rate", f"{win_rate:.1%}")
k4.metric("High Priority Opps", f"{high_priority_pct:.1%}")

st.divider()

seg_colors = alt.Scale(
    domain=["Low", "Medium", "High"],
    range=["#e74c3c", "#f39c12", "#2ecc71"],
)

stage_colors = alt.Scale(
    domain=stage_order,
    range=[
        "#95a5a6",
        "#7f8c8d",
        "#3498db",
        "#9b59b6",
        "#e67e22",
        "#2ecc71",
        "#e74c3c",
        "#c0392b",
    ],
)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown(
        '<p class="section-header">Pipeline by Stage</p>', unsafe_allow_html=True
    )
    pipeline_data = (
        chart_all[chart_all["stage"].isin(active_stages)]
        .groupby("stage")
        .agg(count=("opportunity_id", "count"), total_value=("deal_amount", "sum"))
        .reset_index()
    )
    pipeline_chart = (
        alt.Chart(to_display(pipeline_data))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("stage:N", sort=active_stages, title="Stage"),
            y=alt.Y("count:Q", title="Number of Opportunities"),
            color=alt.Color("stage:N", scale=stage_colors, legend=None),
            tooltip=[
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("count:Q", title="Opportunities"),
                alt.Tooltip("total_value:Q", title="Total Value", format="$,.0f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(pipeline_chart, use_container_width=True)

with c2:
    st.markdown(
        '<p class="section-header">Win Rate by Stage</p>', unsafe_allow_html=True
    )
    win_data = (
        closed.groupby("stage")["is_successful"]
        .mean()
        .reset_index()
        .rename(columns={"is_successful": "win_rate"})
    )
    win_chart = (
        alt.Chart(to_display(win_data))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("stage:N", sort=stage_order, title="Stage"),
            y=alt.Y("win_rate:Q", title="Win Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color("stage:N", scale=stage_colors, legend=None),
            tooltip=[
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("win_rate:Q", title="Win Rate", format=".1%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(win_chart, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown(
        '<p class="section-header">Deal Size by Gift Type</p>', unsafe_allow_html=True
    )
    gift_type_order = ["Annual Giving", "Major Gifts", "Planned Giving"]
    size_data = (
        chart_all.groupby("gift_type")["deal_amount"]
        .median()
        .reset_index()
        .rename(columns={"deal_amount": "median_deal"})
    )
    size_chart = (
        alt.Chart(to_display(size_data))
        .mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#3498db", opacity=0.85
        )
        .encode(
            x=alt.X("gift_type:N", sort=gift_type_order, title="Gift Type"),
            y=alt.Y(
                "median_deal:Q",
                title="Median Deal Size ($)",
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=[
                alt.Tooltip("gift_type:N", title="Gift Type"),
                alt.Tooltip("median_deal:Q", title="Median Deal", format="$,.0f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(size_chart, use_container_width=True)

with c4:
    st.markdown(
        '<p class="section-header">Close Probability vs Deal Size</p>',
        unsafe_allow_html=True,
    )
    scatter_data = chart_open[
        ["deal_amount", "close_probability", "priority", "stage"]
    ].copy()
    scatter_data["deal_amount"] = scatter_data["deal_amount"].astype(float)
    scatter_data["close_probability"] = scatter_data["close_probability"].astype(float)
    sampled = scatter_data.sample(
        min(1000, len(scatter_data)), random_state=42
    ).reset_index(drop=True)

    scatter = (
        alt.Chart(sampled)
        .mark_circle(size=35, opacity=0.45)
        .encode(
            x=alt.X(
                "deal_amount:Q", title="Deal Amount ($)", axis=alt.Axis(format="$,.0f")
            ),
            y=alt.Y(
                "close_probability:Q",
                title="Close Probability",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "priority:N", scale=seg_colors, legend=alt.Legend(title="Priority")
            ),
            tooltip=[
                alt.Tooltip("deal_amount:Q", title="Deal Amount", format="$,.0f"),
                alt.Tooltip(
                    "close_probability:Q", title="Close Probability", format=".1%"
                ),
                alt.Tooltip("priority:N", title="Priority"),
                alt.Tooltip("stage:N", title="Stage"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(scatter, use_container_width=True)

st.divider()

# ── Model Lift Chart ──────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">Model vs Random: How Much Better?</p>',
    unsafe_allow_html=True,
)
st.caption(
    "Contacting top-scored opportunities finds winners faster — here's how much faster."
)

lift_df = closed[
    [
        (
            "close_probability"
            if "close_probability" in closed.columns
            else "stage_probability_pct"
        ),
        "is_successful",
    ]
].copy()

# Re-score closed opps for lift chart
closed_scored = closed.copy()
closed_scored["close_probability"] = model.predict_proba(
    closed_scored[
        [c + "_enc" for c in cat_cols] + ["stage_probability_pct", "deal_amount"]
    ]
)[:, 1]
lift_df = closed_scored[["close_probability", "is_successful"]].copy()
lift_df = lift_df.sort_values("close_probability", ascending=False).reset_index(
    drop=True
)

total_l = len(lift_df)
total_won = lift_df["is_successful"].sum()

steps = list(range(1, 101))
model_capture, random_capture = [], []
for pct in steps:
    n = max(1, int(total_l * pct / 100))
    captured = lift_df["is_successful"].iloc[:n].sum()
    model_capture.append(captured / total_won * 100)
    random_capture.append(pct)

lift_data = pd.DataFrame(
    {
        "pct_contacted": steps + steps,
        "pct_won_captured": model_capture + random_capture,
        "type": ["Model"] * 100 + ["Random"] * 100,
    }
)

lift_chart = (
    alt.Chart(lift_data)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("pct_contacted:Q", title="% of Opportunities Contacted"),
        y=alt.Y("pct_won_captured:Q", title="% of Won Deals Captured"),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["Model", "Random"], range=["#2ecc71", "#aaaaaa"]),
            legend=alt.Legend(title="", orient="bottom-right"),
        ),
        strokeDash=alt.condition(
            alt.datum.type == "Random",
            alt.value([4, 4]),
            alt.value([0]),
        ),
        tooltip=[
            alt.Tooltip("pct_contacted:Q", title="% Contacted"),
            alt.Tooltip("pct_won_captured:Q", title="% Won Captured", format=".1f"),
            alt.Tooltip("type:N", title="Method"),
        ],
    )
)

top33_model = (
    lift_df["is_successful"].iloc[: int(total_l * 0.33)].sum() / total_won * 100
)
lift_at_33 = top33_model / 33.0

col_chart, col_stat = st.columns([8, 1])
with col_chart:
    st.altair_chart(lift_chart.properties(height=350), use_container_width=True)
with col_stat:
    st.metric(
        "Lift at top 33%",
        f"{lift_at_33:.1f}x",
        help="Scoring top 33% of opportunities captures this many times more winners than random.",
    )
    st.metric(
        "Won deals captured (top 33%)",
        f"{top33_model:.1f}%",
        help="% of all won deals found by prioritizing only the top third.",
    )
    st.metric(
        "Model AUC",
        f"{auc:.2f}",
        help="Area under ROC curve. 0.5 = random, 1.0 = perfect.",
    )

st.divider()

# ── Top Opportunities Leaderboard ─────────────────────────────────────────────
display_df = chart_open.sort_values("close_probability", ascending=False).reset_index(
    drop=True
)

st.markdown(
    f'<p class="section-header">Top Opportunities '
    f'<span style="font-weight:400;color:#888;">({len(display_df):,} open · {gift_filter})</span></p>',
    unsafe_allow_html=True,
)

show_cols = [
    "opportunity_id",
    "donor_id",
    "stage",
    "gift_type",
    "deal_amount",
    "stage_probability_pct",
    "close_probability",
    "priority",
    "stage_entry_date",
    "fiscal_year",
    "fiscal_quarter",
]

st.dataframe(
    to_display(display_df[show_cols]),
    use_container_width=True,
    height=350,
    hide_index=True,
    column_config={
        "opportunity_id": st.column_config.TextColumn("Opportunity ID"),
        "donor_id": st.column_config.TextColumn("Donor ID"),
        "stage": st.column_config.TextColumn("Stage"),
        "gift_type": st.column_config.TextColumn("Gift Type"),
        "deal_amount": st.column_config.NumberColumn("Deal Amount", format="$%,.0f"),
        "stage_probability_pct": st.column_config.NumberColumn(
            "Stage %", format="%d%%"
        ),
        "close_probability": st.column_config.ProgressColumn(
            "Close Probability",
            min_value=0,
            max_value=1,
            format="%.2f",
        ),
        "priority": st.column_config.TextColumn("Priority"),
        "stage_entry_date": st.column_config.TextColumn("Stage Entry Date"),
        "fiscal_year": st.column_config.NumberColumn("FY", format="%d"),
        "fiscal_quarter": st.column_config.NumberColumn("Q", format="%d"),
    },
)

st.divider()

# ── Download ──────────────────────────────────────────────────────────────────
csv = display_df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download opportunities",
    data=csv,
    file_name="opportunity_scores.csv",
    mime="text/csv",
)
