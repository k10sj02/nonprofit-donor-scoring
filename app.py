import streamlit as st

st.set_page_config(
    page_title="Donor Intelligence Suite",
    page_icon="💚",
    layout="wide",
)

st.title("💚 Donor Intelligence Suite")
st.caption("A lightweight framework for donor retention and pipeline scoring in nonprofit fundraising.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧠 Donor Propensity Scoring")
    st.markdown("""
    Predict which donors are most likely to give again using historical donation behavior.
    - RFM-based feature engineering
    - Logistic regression propensity model
    - Segment donors into Low / Medium / High tiers
    - Model lift vs random selection
    """)
    st.page_link("pages/1_Donor_Propensity.py", label="Open Donor Propensity →", icon="🧠")

with col2:
    st.markdown("### 🎯 Opportunity Scoring")
    st.markdown("""
    Score and prioritize open pipeline opportunities by likelihood to close.
    - Pipeline overview by stage and gift type
    - ML model trained on closed opportunities
    - Prioritize High / Medium / Low opportunities
    - Model lift vs random selection
    """)
    st.page_link("pages/2_Opportunity_Scoring.py", label="Open Opportunity Scoring →", icon="🎯")

st.divider()
st.caption("Built with Streamlit · Data generated synthetically for demonstration purposes.")
