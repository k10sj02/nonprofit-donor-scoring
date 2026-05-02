import streamlit as st

st.set_page_config(
    page_title="Donor Intelligence Suite",
    page_icon="💚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 4rem; }
    .hero { text-align: center; padding: 2rem 0 1rem 0; }
    .hero h1 { font-size: 2.8rem; font-weight: 700; }
    .hero p { font-size: 1.1rem; color: #666; max-width: 600px; margin: 0 auto; }
    .feature-box { background: #f8f9fa; border-radius: 12px; padding: 1.75rem; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <div style="font-size:3rem">💚</div>
    <h1>Donor Intelligence Suite</h1>
    <p>Score, prioritize, and maximize donor value using behavioral data and predictive modeling.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

_, col, _ = st.columns([1, 2, 1])
with col:
    st.markdown(
        """
    <div class="feature-box">
        <h3>🧠 Donor Propensity & Value Scoring</h3>
        <p>Identify who to contact now and who to invest in long-term using predictive scoring and value modeling.</p>
        <ul>
            <li>RFM-based feature engineering</li>
            <li>Random Forest propensity model</li>
            <li>Expected Next Gift (likelihood × donation size)</li>
            <li>Predicted LTV for long-term prioritization</li>
            <li>Segment donors into four tiers</li>
            <li>Model lift vs random selection</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption(
        "Balances short-term fundraising opportunities with long-term donor value."
    )

    st.page_link(
        "pages/1_Donor_Propensity.py", label="Open Donor Propensity →", icon="🧠"
    )

st.divider()

_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.caption(
        "Built by Stann-Omar Jones · Upload your own donation CSV to get started."
    )
