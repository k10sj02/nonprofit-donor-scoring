import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
    .block-container { padding-top: 2rem; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4a4a4a;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """Converts to Arrow-safe dtypes for Streamlit. Formats datetimes as strings."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return pd.DataFrame(out.to_dict("records"))


# ── Header ────────────────────────────────────────────────────────────────────
st.title("💚 Donor Propensity Scoring")
st.caption(
    "Predict which donors are most likely to give again using historical donation behavior."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload donation data (CSV)", type="csv")

    PROJECT_ROOT = Path(__file__).parent
    default_path = PROJECT_ROOT / "outputs" / "donations_clean.csv"

    df = None
    if uploaded_file:
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
        options=["All", "High", "Medium", "Low"],
        label_visibility="collapsed",
    )

if df is None:
    st.info("👈 Upload a dataset in the sidebar to get started.")
    st.stop()

# ── Data prep & model ─────────────────────────────────────────────────────────
df["donation_date"] = pd.to_datetime(df["donation_date"])

cutoff_date = df["donation_date"].quantile(0.8)
train_df = df[df["donation_date"] <= cutoff_date]
future_df = df[df["donation_date"] > cutoff_date]

donor_summary = (
    train_df.groupby("donor_id")
    .agg(
        donation_count=("donation_id", "count"),
        total_donated=("donation_amount", "sum"),
        last_donation=("donation_date", "max"),
    )
    .reset_index()
)

donor_summary["recency_days"] = (cutoff_date - donor_summary["last_donation"]).dt.days
donor_summary["avg_donation"] = (
    donor_summary["total_donated"] / donor_summary["donation_count"]
)

future_donors = future_df["donor_id"].unique()
donor_summary["donated_again"] = (
    donor_summary["donor_id"].isin(future_donors).astype(int)
)

features = ["donation_count", "recency_days", "total_donated", "avg_donation"]
X = donor_summary[features]
y = donor_summary["donated_again"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

donor_summary["propensity_score"] = model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, donor_summary["propensity_score"])

donor_summary["segment"] = pd.qcut(
    donor_summary["propensity_score"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop",
).astype(str)

# ── Apply segment filter to a view used by all charts and table ───────────────
if segment_filter != "All":
    chart_df = donor_summary[donor_summary["segment"] == segment_filter].copy()
else:
    chart_df = donor_summary.copy()

# ── KPIs — always based on full dataset ───────────────────────────────────────
total_donors = len(donor_summary)
high_pct = (donor_summary["segment"] == "High").mean()
repeat_rate = (donor_summary["donated_again"] == 1).mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Donors", f"{total_donors:,}")
k2.metric("High Propensity", f"{high_pct:.1%}")
k3.metric("Model AUC", f"{auc:.2f}")
k4.metric("Overall Retention Rate", f"{repeat_rate:.1%}")

st.divider()

seg_color_scale = alt.Scale(
    domain=["Low", "Medium", "High"],
    range=["#e74c3c", "#f39c12", "#2ecc71"],
)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown(
        '<p class="section-header">Propensity Score Distribution</p>',
        unsafe_allow_html=True,
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
    st.markdown(
        '<p class="section-header">Retention Rate by Segment</p>',
        unsafe_allow_html=True,
    )
    segment_perf = (
        donor_summary.groupby("segment", observed=True)["donated_again"]
        .mean()
        .reset_index()
        .rename(columns={"donated_again": "retention_rate"})
    )

    bar_chart = (
        alt.Chart(to_display(segment_perf))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("segment:N", sort=["Low", "Medium", "High"], title="Segment"),
            y=alt.Y(
                "retention_rate:Q", title="Retention Rate", axis=alt.Axis(format=".0%")
            ),
            color=alt.Color("segment:N", scale=seg_color_scale, legend=None),
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
        '<p class="section-header">Donations per Donor</p>', unsafe_allow_html=True
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
        '<p class="section-header">Avg Donation vs Recency by Segment</p>',
        unsafe_allow_html=True,
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
                "segment:N", scale=seg_color_scale, legend=alt.Legend(title="Segment")
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
            color=alt.Color("segment:N", scale=seg_color_scale, legend=None),
            detail="segment:N",
        )
    )

    coef_all = np.polyfit(sampled["recency_days"], sampled["avg_donation"], 1)
    x0g, x1g = int(sampled["recency_days"].min()), int(sampled["recency_days"].max())
    global_df = pd.DataFrame(
        [
            {"x": x0g, "y": float(coef_all[0] * x0g + coef_all[1])},
            {"x": x1g, "y": float(coef_all[0] * x1g + coef_all[1])},
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

# ── Donor table ───────────────────────────────────────────────────────────────
display_df = chart_df.sort_values("propensity_score", ascending=False).reset_index(
    drop=True
)

st.markdown(
    f'<p class="section-header">Top Donors '
    f'<span style="font-weight:400;color:#888;">({len(display_df):,} donors · {segment_filter} segment)</span></p>',
    unsafe_allow_html=True,
)

st.dataframe(
    to_display(display_df),
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
            "Propensity Score",
            min_value=0,
            max_value=1,
            format="%.2f",
        ),
        "segment": st.column_config.TextColumn("Segment"),
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
