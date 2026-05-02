import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Upload & Map Data",
    page_icon="📂",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4a4a4a;
        margin-bottom: 0.25rem;
    }
    .field-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2c3e50;
    }
    .success-box {
        background: #eafaf1;
        border-left: 4px solid #2ecc71;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .error-box {
        background: #fdedec;
        border-left: 4px solid #e74c3c;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Required fields and their descriptions ────────────────────────────────────
REQUIRED_FIELDS = {
    "donor_id": "Unique identifier per donor (e.g. DonorID, donor_id, ContactID)",
    "donation_id": "Unique identifier per transaction (e.g. GiftID, transaction_id)",
    "donation_date": "Date of the donation (e.g. GiftDate, donation_date, Date)",
    "donation_amount": "Numeric gift amount (e.g. Amount, GiftAmount, donation_amount)",
}

OPTIONAL_FIELDS = {
    "donation_type": "Type of gift e.g. One-time, Recurring (optional)",
    "campaign": "Campaign or fund name (optional)",
    "payment_method": "Payment method e.g. Credit Card, Check (optional)",
}

# ── Fuzzy column matcher ──────────────────────────────────────────────────────
FIELD_HINTS = {
    "donor_id": [
        "donor_id",
        "donorid",
        "donor",
        "contactid",
        "contact_id",
        "id",
        "constituent_id",
    ],
    "donation_id": [
        "donation_id",
        "donationid",
        "gift_id",
        "giftid",
        "transaction_id",
        "transactionid",
        "record_id",
    ],
    "donation_date": [
        "donation_date",
        "donationdate",
        "gift_date",
        "giftdate",
        "date",
        "transaction_date",
        "close_date",
    ],
    "donation_amount": [
        "donation_amount",
        "amount",
        "gift_amount",
        "giftamount",
        "value",
        "gift_value",
        "total",
    ],
    "donation_type": ["donation_type", "gift_type", "type", "giving_type"],
    "campaign": ["campaign", "fund", "appeal", "designation"],
    "payment_method": ["payment_method", "paymentmethod", "payment_type", "method"],
}


def best_guess(field: str, columns: list[str]) -> str:
    """Return the best matching column name for a required field, or empty string."""
    hints = FIELD_HINTS.get(field, [])
    col_lower = {c.lower().replace(" ", "_"): c for c in columns}
    for hint in hints:
        if hint in col_lower:
            return col_lower[hint]
    return ""


def validate_mapping(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Validate the mapped columns. Returns a dict of field -> (ok, message).
    """
    results = {}
    for field, col in mapping.items():
        if field not in REQUIRED_FIELDS:
            continue
        if not col or col == "(not mapped)":
            results[field] = (False, "Required — please select a column.")
            continue

        series = df[col]

        if field == "donation_date":
            parsed = pd.to_datetime(series, errors="coerce")
            n_failed = parsed.isna().sum()
            if n_failed == len(series):
                results[field] = (
                    False,
                    f"Could not parse any dates in '{col}'. Check the format.",
                )
            elif n_failed > 0:
                results[field] = (
                    True,
                    f"⚠️ {n_failed:,} rows could not be parsed as dates and will be dropped.",
                )
            else:
                results[field] = (
                    True,
                    f"✅ All {len(series):,} dates parsed successfully.",
                )

        elif field == "donation_amount":
            numeric = pd.to_numeric(series, errors="coerce")
            n_failed = numeric.isna().sum()
            if n_failed == len(series):
                results[field] = (
                    False,
                    f"Could not parse any numeric values in '{col}'.",
                )
            elif n_failed > 0:
                results[field] = (
                    True,
                    f"⚠️ {n_failed:,} rows could not be parsed as numbers and will be dropped.",
                )
            else:
                results[field] = (True, f"✅ All {len(series):,} values are numeric.")

        elif field in ("donor_id", "donation_id"):
            n_null = series.isna().sum()
            n_dupes = series.duplicated().sum() if field == "donation_id" else 0
            msg = f"✅ {series.nunique():,} unique values"
            if n_null > 0:
                msg += f" · ⚠️ {n_null:,} nulls found"
            if n_dupes > 0 and field == "donation_id":
                msg += f" · ⚠️ {n_dupes:,} duplicate IDs found"
            results[field] = (True, msg)

    return results


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename columns per mapping, parse dates and amounts, drop unmapped optional cols.
    """
    rename_map = {v: k for k, v in mapping.items() if v and v != "(not mapped)"}
    out = df.rename(columns=rename_map)

    # Keep only mapped columns
    keep = [k for k, v in mapping.items() if v and v != "(not mapped)"]
    out = out[keep].copy()

    # Parse types
    out["donation_date"] = pd.to_datetime(out["donation_date"], errors="coerce")
    out["donation_amount"] = pd.to_numeric(out["donation_amount"], errors="coerce")

    # Drop rows where required fields failed to parse
    out = out.dropna(subset=["donation_date", "donation_amount", "donor_id"])

    return out


# ── Page header ───────────────────────────────────────────────────────────────
st.title("📂 Upload & Map Your Data")
st.caption(
    "Upload your donation CSV, map your columns to the required fields, and proceed to analysis."
)
st.divider()

with st.expander("📋 What format does my CSV need to be in?", expanded=False):
    st.markdown("""
    Your CSV needs at minimum these four fields — the column names don't matter,
    you'll map them in the next step:

    | Required field | What it represents | Example column names |
    |---|---|---|
    | Donor ID | Unique identifier per donor | `DonorID`, `ContactID`, `donor_id` |
    | Transaction ID | Unique identifier per gift | `GiftID`, `TransactionID`, `donation_id` |
    | Gift date | Date of the donation | `GiftDate`, `Date`, `donation_date` |
    | Gift amount | Numeric dollar amount | `Amount`, `GiftAmount`, `donation_amount` |

    Each row should represent one donation transaction.
    One donor can appear on multiple rows (one per gift).
    Dates can be in most common formats — the tool will detect them automatically.
    """)

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">Step 1 — Upload your CSV</p>', unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type="csv", label_visibility="collapsed")

if uploaded_file is None:
    # Offer default dataset as fallback
    PROJECT_ROOT = Path(__file__).parent.parent
    default_path = PROJECT_ROOT / "outputs" / "donations_clean.csv"
    if default_path.exists():
        st.info(
            "No file uploaded — you can use the default sample dataset below, or upload your own."
        )
        if st.button("Use default sample dataset"):
            raw_df = pd.read_csv(default_path)
            st.session_state["raw_df"] = raw_df
            st.session_state["used_default"] = True
    else:
        st.info("Upload a CSV file to get started.")
    if "raw_df" not in st.session_state:
        st.stop()
else:
    raw_df = pd.read_csv(uploaded_file)
    st.session_state["raw_df"] = raw_df
    st.session_state["used_default"] = False

raw_df = st.session_state["raw_df"]

st.success(f"Loaded {len(raw_df):,} rows × {len(raw_df.columns)} columns")
st.markdown("**Preview of your data:**")
st.dataframe(raw_df.head(5), use_container_width=True, hide_index=True)

st.divider()

# ── Step 2: Map columns ───────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">Step 2 — Map your columns</p>', unsafe_allow_html=True
)
st.caption("We've made our best guess at the mapping. Correct any that look wrong.")

columns = raw_df.columns.tolist()
none_option = "(not mapped)"
col_options = [none_option] + columns

mapping = {}

req_col1, req_col2 = st.columns(2)

field_items = list(REQUIRED_FIELDS.items())
for i, (field, desc) in enumerate(field_items):
    guess = best_guess(field, columns)
    col = req_col1 if i % 2 == 0 else req_col2
    with col:
        st.markdown(
            f'<p class="field-label">{"⭐ " + field.replace("_", " ").title()}</p>',
            unsafe_allow_html=True,
        )
        st.caption(desc)
        default_idx = col_options.index(guess) if guess in col_options else 0
        mapping[field] = st.selectbox(
            label=field,
            options=col_options,
            index=default_idx,
            key=f"map_{field}",
            label_visibility="collapsed",
        )

st.markdown("**Optional fields** — map these if available for richer analysis:")
opt_col1, opt_col2, opt_col3 = st.columns(3)
opt_cols_ui = [opt_col1, opt_col2, opt_col3]

for i, (field, desc) in enumerate(OPTIONAL_FIELDS.items()):
    guess = best_guess(field, columns)
    with opt_cols_ui[i % 3]:
        st.markdown(
            f'<p class="field-label">{field.replace("_", " ").title()}</p>',
            unsafe_allow_html=True,
        )
        st.caption(desc)
        default_idx = col_options.index(guess) if guess in col_options else 0
        mapping[field] = st.selectbox(
            label=field,
            options=col_options,
            index=default_idx,
            key=f"map_{field}",
            label_visibility="collapsed",
        )

st.divider()

# ── Step 3: Validate ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Step 3 — Validate</p>', unsafe_allow_html=True)

validation = validate_mapping(raw_df, mapping)
all_ok = all(ok for ok, _ in validation.values())

v1, v2 = st.columns(2)
for i, (field, (ok, msg)) in enumerate(validation.items()):
    col = v1 if i % 2 == 0 else v2
    with col:
        icon = "✅" if ok else "❌"
        color = "#2ecc71" if ok else "#e74c3c"
        st.markdown(
            f'<div style="border-left: 4px solid {color}; padding: 0.5rem 0.75rem; '
            f'margin-bottom: 0.5rem; border-radius: 4px; background: {"#eafaf1" if ok else "#fdedec"};">'
            f'<strong>{icon} {field.replace("_", " ").title()}</strong><br>'
            f'<span style="font-size:0.85rem">{msg}</span></div>',
            unsafe_allow_html=True,
        )

if not all_ok:
    st.warning("Fix the errors above before proceeding.")
    st.stop()

st.divider()

# ── Step 4: Preview transformed data ─────────────────────────────────────────
st.markdown(
    '<p class="section-header">Step 4 — Preview transformed data</p>',
    unsafe_allow_html=True,
)

transformed = apply_mapping(raw_df, mapping)

c1, c2, c3 = st.columns(3)
c1.metric("Rows after transformation", f"{len(transformed):,}")
c2.metric("Donors", f"{transformed['donor_id'].nunique():,}")
c3.metric(
    "Date range",
    f"{transformed['donation_date'].min().strftime('%Y-%m-%d')} → "
    f"{transformed['donation_date'].max().strftime('%Y-%m-%d')}",
)

st.dataframe(transformed.head(10), use_container_width=True, hide_index=True)

n_dropped = len(raw_df) - len(transformed)
if n_dropped > 0:
    st.warning(f"{n_dropped:,} rows were dropped due to unparseable dates or amounts.")

st.divider()

# ── Step 5: Confirm and proceed ───────────────────────────────────────────────
st.markdown(
    '<p class="section-header">Step 5 — Confirm and analyze</p>', unsafe_allow_html=True
)
st.caption(
    "This will save your mapped data and make it available to the Donor Propensity dashboard."
)

if st.button("✅ Confirm and go to Donor Propensity →", type="primary"):
    st.session_state["mapped_df"] = transformed
    st.session_state["mapping_done"] = True
    st.switch_page("pages/1_Donor_Propensity.py")

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
