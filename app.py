import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through records to guarantee numpy-native dtypes.
    Eliminates LargeUtf8, Categorical, and all pandas extension types
    before Streamlit hands the DataFrame to PyArrow."""
    return pd.DataFrame(df.to_dict("records"))


st.title("Donor Propensity Scoring Tool")

st.write(
    "This tool predicts which donors are most likely to give again using historical donation behavior."
)

uploaded_file = st.file_uploader("Upload donation data (CSV)")

# Anchor default path to the project root (same folder as app.py),
# regardless of which directory Streamlit is launched from.
PROJECT_ROOT = Path(__file__).parent
default_path = PROJECT_ROOT / "outputs" / "donations_clean.csv"

# ✅ LOAD DATA
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset")

elif default_path.exists():
    df = pd.read_csv(default_path)
    st.info(f"Using default dataset ({default_path.relative_to(PROJECT_ROOT)})")

else:
    st.warning(
        "No dataset found. Please upload a file or run the notebook first to generate outputs/donations_clean.csv."
    )
    st.stop()

# 🔥 EVERYTHING BELOW RUNS ONLY IF df EXISTS

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

# 🔥 MODEL PERFORMANCE
auc = roc_auc_score(y, donor_summary["propensity_score"])

donor_summary["segment"] = pd.qcut(
    donor_summary["propensity_score"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop",
)

# 🔥 KPI METRICS
total_donors = len(donor_summary)
high_pct = (donor_summary["segment"] == "High").mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total donors", total_donors)

with col2:
    st.metric("High propensity donors", f"{high_pct:.1%}")

with col3:
    st.metric("Model AUC", f"{auc:.2f}")

# 🔥 FILTER
segment_filter = st.selectbox("Filter by segment", ["All", "High", "Medium", "Low"])

if segment_filter != "All":
    filtered_df = donor_summary[donor_summary["segment"] == segment_filter]
else:
    filtered_df = donor_summary

# 🔥 TOP DONORS TABLE
st.subheader("Top Donors")

display_df = filtered_df.sort_values("propensity_score", ascending=False).reset_index(
    drop=True
)

st.dataframe(to_display(display_df))

# 🔥 SEGMENT PERFORMANCE
st.subheader("Segment Performance")

segment_perf = (
    donor_summary.groupby("segment", observed=True)["donated_again"]
    .mean()
    .reset_index()
    .rename(columns={"donated_again": "retention_rate"})
)

st.dataframe(to_display(segment_perf))

# 🔥 DOWNLOAD BUTTON
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download results",
    data=csv,
    file_name="donor_propensity.csv",
    mime="text/csv",
)
