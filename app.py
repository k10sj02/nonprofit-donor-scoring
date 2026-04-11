import streamlit as st

st.set_page_config(
    page_title="Donor Intelligence Suite",
    page_icon="💚",
    layout="wide",
)

st.markdown("""
    <style>
    .block-container { padding-top: 4rem; }
    .hero { text-align: center; padding: 2rem 0 1rem 0; }
    .hero h1 { font-size: 2.8rem; font-weight: 700; }
    .hero p { font-size: 1.1rem; color: #666; max-width: 600px; margin: 0 auto; }
    .feature-box { background: #f8f9fa; border-radius: 12px; padding: 1.75rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div style="font-size:3rem">💚</div>
    <h1>Donor Intelligence Suite</h1>
    <p>A lightweight, open source framework for donor retention and propensity scoring
    in nonprofit fundraising.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

_, col, _ = st.columns([1, 2, 1])
with col:
    st.markdown("""
    <div class="feature-box">
        <h3>🧠 Donor Propensity Scoring</h3>
        <p>Predict which donors are most likely to give again using historical donation behavior.</p>
        <ul>
            <li>RFM-based feature engineering</li>
            <li>Random Forest propensity model</li>
            <li>Segment donors into four propensity tiers</li>
            <li>Model lift vs random selection</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.page_link("pages/1_Donor_Propensity.py", label="Open Donor Propensity →", icon="🧠")

st.divider()
_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.caption("Built with Streamlit · Upload your own donation CSV to get started.")