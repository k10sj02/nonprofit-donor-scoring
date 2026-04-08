import streamlit as st

st.set_page_config(
    page_title="Donor Intelligence Suite",
    page_icon="💚",
    layout="wide",
)

st.title("💚 Donor Intelligence Suite")
st.caption("A lightweight framework for donor retention and propensity scoring in nonprofit fundraising.")
st.divider()

st.markdown("### 🧠 Donor Propensity Scoring")
st.markdown("""
Predict which donors are most likely to give again using historical donation behavior.
- RFM-based feature engineering
- Logistic regression propensity model
- Segment donors into Low / Medium / High tiers
- Model lift vs random selection
""")
st.page_link("pages/1_Donor_Propensity.py", label="Open Donor Propensity →", icon="🧠")

st.divider()
st.caption("Built with Streamlit · Data generated synthetically for demonstration purposes.")