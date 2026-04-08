import pandas as pd
import numpy as np
import random
import os
import uuid
from datetime import datetime, timedelta


# -----------------------------
# HELPERS
# -----------------------------
def random_date(start_year=2019, end_year=2025):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def stage_probability(stage):
    mapping = {
        "Discovery":   random.randint(5, 20),
        "Engagement":  random.randint(15, 35),
        "Evaluation":  random.randint(30, 55),
        "Proposal":    random.randint(50, 75),
        "Negotiation": random.randint(70, 90),
        "Committed":   100,
        "Lost":        0,
        "Rejected":    0,
    }
    return mapping[stage]


def base_win_prob(gift_type: str, amount: float) -> float:
    """
    Base win probability driven by gift type and deal size.
    This creates realistic feature-outcome correlations so the ML model
    has something meaningful to learn beyond just the terminal stage.

    Annual Giving closes more often (smaller, simpler asks).
    Planned Giving closes less often (complex, longer cycle).
    Larger deals are harder to close regardless of type.
    """
    base = {
        "Annual Giving":  0.55,
        "Major Gifts":    0.38,
        "Planned Giving": 0.22,
    }.get(gift_type, 0.38)

    # Deal size modifier
    if amount > 200_000:
        base *= 0.60
    elif amount > 50_000:
        base *= 0.80
    elif amount < 5_000:
        base *= 1.20

    return float(np.clip(base, 0.05, 0.95))


def assign_outcome(stage, close_date, late_stages, win_prob: float):
    """
    Assign outcome with:
    - Feature-correlated win probability (not purely stage-driven)
    - Noise on terminal stages so they're not perfectly deterministic
    - Future dates = still open
    """
    today = datetime.today()

    if close_date > today:
        return False, False

    if stage in ["Lost", "Rejected"]:
        # Mostly lost, but small revival chance
        return True, random.random() < 0.04

    if stage == "Committed":
        # Mostly won, but small fall-through chance
        return True, random.random() < 0.88

    if stage in late_stages:
        # Late stage — blend stage signal with feature-driven probability
        blended = 0.5 * 0.78 + 0.5 * win_prob
        return True, random.random() < blended

    # Early/mid stage — mostly feature-driven
    return True, random.random() < win_prob


def assign_gift_type(amount: int) -> str:
    if amount < 5_000:
        return "Annual Giving"
    elif amount < 25_000:
        return "Major Gifts"
    else:
        return "Planned Giving"


def synthetic_ask_amount() -> int:
    """
    Draw from three tiers directly to ensure meaningful gift type distribution.
    ~40% Annual, ~45% Major, ~15% Planned.
    """
    tier = random.choices(
        ["annual", "major", "planned"],
        weights=[40, 45, 15],
        k=1,
    )[0]

    if tier == "annual":
        return random.randint(1_000, 4_999)
    elif tier == "major":
        return random.randint(5_000, 24_999)
    else:
        return random.randint(25_000, 500_000)


# Stage progression order
STAGE_PROGRESSION = [
    "Discovery",
    "Engagement",
    "Evaluation",
    "Proposal",
    "Negotiation",
    "Committed",
]


def simulate_lifecycle(start_date: datetime, win_prob: float) -> list[dict]:
    """
    Simulate a gift's journey through pipeline stages.
    win_prob influences how far opportunities progress and whether they close won.
    Higher win_prob = more likely to advance stages and close successfully.
    """
    transitions = []
    current_date = start_date

    # Entry stage — higher win_prob opportunities tend to enter later
    # (they've already been qualified)
    entry_weights = [40, 25, 15, 10, 7, 3]
    if win_prob > 0.5:
        entry_weights = [20, 20, 20, 20, 15, 5]

    entry_idx = random.choices(range(len(STAGE_PROGRESSION)), weights=entry_weights, k=1)[0]
    current_stage  = STAGE_PROGRESSION[entry_idx]
    previous_stage = None

    while True:
        transitions.append({
            "stage":                 current_stage,
            "previous_stage":        previous_stage,
            "stage_entry_date":      current_date.strftime("%Y-%m-%d"),
            "stage_probability_pct": stage_probability(current_stage),
            "is_current":            False,
        })

        days_in_stage = random.randint(14, 120)
        current_date  = current_date + timedelta(days=days_in_stage)

        stage_idx    = STAGE_PROGRESSION.index(current_stage)
        is_last_stage = (stage_idx == len(STAGE_PROGRESSION) - 1)

        if is_last_stage:
            # Already at Committed — stop
            break

        roll = random.random()

        # Advancement probability influenced by win_prob
        advance_prob = 0.45 + (win_prob * 0.25)  # range: ~0.46 - 0.70
        stall_prob   = advance_prob + 0.15

        if roll < advance_prob:
            previous_stage = current_stage
            current_stage  = STAGE_PROGRESSION[stage_idx + 1]
        elif roll < stall_prob:
            terminal = random.choice(["Lost", "Rejected"])
            transitions.append({
                "stage":                 terminal,
                "previous_stage":        current_stage,
                "stage_entry_date":      current_date.strftime("%Y-%m-%d"),
                "stage_probability_pct": 0,
                "is_current":            False,
            })
            break
        else:
            # Still open
            break

    if transitions:
        transitions[-1]["is_current"] = True

    return transitions


def is_closed_won(transitions: list[dict], win_prob: float) -> tuple[bool, bool]:
    """Derive is_closed and is_successful with noise on terminal stages."""
    last_stage = transitions[-1]["stage"]
    late_stages = ["Negotiation"]

    if last_stage == "Committed":
        return True, random.random() < 0.88
    elif last_stage in ["Lost", "Rejected"]:
        return True, random.random() < 0.04
    elif last_stage in late_stages:
        blended = 0.5 * 0.78 + 0.5 * win_prob
        # Only close if past date — keep open otherwise
        return False, False
    else:
        return False, False


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_opportunities():

    print("🔄 Starting generation...")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    donors = pd.read_csv("data/raw/donor_profiles.csv")
    gifts  = pd.read_csv("data/raw/donation_records.csv")

    # -----------------------------
    # DETECT CAPACITY COLUMN
    # -----------------------------
    capacity_col = None
    for col in donors.columns:
        if "capacity" in col.lower() or "wealth" in col.lower():
            capacity_col = col
            break

    capacity_options = [
        "50K-100K", "100K-250K", "250K-500K",
        "500K-1M",  "1M-2M",    "2M-5M",    "5M+",
    ]

    # -----------------------------
    # DERIVE DONOR FEATURES
    # -----------------------------
    if "donor_id" in gifts.columns:
        numeric_cols = gifts.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "donor_id"]
        preferred    = [c for c in numeric_cols if "amount" in c.lower() or "value" in c.lower()]
        amount_col   = preferred[0] if preferred else (numeric_cols[0] if numeric_cols else None)

        if amount_col:
            print(f"💰 Using '{amount_col}' as donation column")
            donor_stats = (
                gifts.groupby("donor_id")
                .agg(avg_gift=(amount_col, "mean"), gift_count=(amount_col, "count"))
                .reset_index()
            )
            donors = donors.merge(donor_stats, on="donor_id", how="left")

    donors["avg_gift"]   = donors.get("avg_gift",   pd.Series()).fillna(500)
    donors["gift_count"] = donors.get("gift_count", pd.Series()).fillna(1)

    # -----------------------------
    # STABLE CONTACT ID MAP (1:1 with donor_id)
    # -----------------------------
    contact_id_map = {
        donor_id: generate_id("contact")
        for donor_id in donors["donor_id"].unique()
    }

    # -----------------------------
    # GENERATE OPPORTUNITIES + STAGE HISTORY
    # -----------------------------
    AVG_OPPS_PER_DONOR = 3.5
    MAX_OPPS_PER_DONOR = 10

    opp_rows   = []
    stage_rows = []

    for _, donor in donors.iterrows():

        donor_id   = donor["donor_id"]
        contact_id = contact_id_map[donor_id]
        sector     = donor.get("sector", "Unknown")

        num_opps = min(np.random.poisson(AVG_OPPS_PER_DONOR), MAX_OPPS_PER_DONOR)
        num_opps = max(num_opps, 1)

        for _ in range(num_opps):

            opp_id     = generate_id("opp")
            start_date = random_date()
            amount     = synthetic_ask_amount()
            gift_type  = assign_gift_type(amount)

            # Compute feature-driven win probability for this opportunity
            win_prob = base_win_prob(gift_type, amount)

            # Add sector modifier
            sector_boost = {
                "Healthcare":   0.05,
                "Technology":   0.03,
                "Government":  -0.05,
                "Education":    0.02,
            }.get(sector, 0.0)
            win_prob = float(np.clip(win_prob + sector_boost, 0.05, 0.95))

            fiscal_year    = start_date.year + (1 if start_date.month >= 7 else 0)
            fiscal_quarter = ((start_date.month - 1) // 3) + 1

            capacity_value = (
                donor.get(capacity_col)
                if capacity_col and pd.notna(donor.get(capacity_col))
                else random.choice(capacity_options)
            )

            transitions = simulate_lifecycle(start_date, win_prob)
            is_closed, is_successful = is_closed_won(transitions, win_prob)

            current = transitions[-1]

            opp_rows.append({
                "opportunity_id":                opp_id,
                "donor_id":                      donor_id,
                "contact_id":                    contact_id,
                "sector":                        sector,
                "capacity_band":                 capacity_value,
                "gift_type":                     gift_type,
                "deal_amount":                   amount,
                "current_stage":                 current["stage"],
                "current_stage_probability_pct": current["stage_probability_pct"],
                "stage_entry_date":              current["stage_entry_date"],
                "fiscal_year":                   fiscal_year,
                "fiscal_quarter":                fiscal_quarter,
                "is_closed":                     is_closed,
                "is_successful":                 is_successful,
                "total_stage_count":             len(transitions),
                "win_prob_signal":               round(win_prob, 4),
            })

            for t in transitions:
                stage_rows.append({
                    "opportunity_id":        opp_id,
                    "donor_id":              donor_id,
                    "stage":                 t["stage"],
                    "previous_stage":        t["previous_stage"],
                    "stage_entry_date":      t["stage_entry_date"],
                    "stage_probability_pct": t["stage_probability_pct"],
                    "is_current":            t["is_current"],
                    "deal_amount":           amount,
                    "gift_type":             gift_type,
                    "sector":                sector,
                    "win_prob_signal":       round(win_prob, 4),
                })

    opps   = pd.DataFrame(opp_rows)
    stages = pd.DataFrame(stage_rows)

    # -----------------------------
    # SAVE
    # -----------------------------
    os.makedirs("data/synthetic", exist_ok=True)

    opps.to_csv("data/synthetic/synthetic_opportunities.csv",   index=False)
    stages.to_csv("data/synthetic/opportunity_stages.csv", index=False)

    print(f"\n✅ Opportunities generated:   {len(opps):,}")
    print(f"📋 Stage transitions logged:  {len(stages):,}")
    print(f"📊 Avg opps per donor:        {len(opps) / len(donors):.1f}")
    print(f"\n🎁 Gift type distribution:")
    print(opps["gift_type"].value_counts().to_string())
    print(f"\n🏁 Outcome distribution:")
    print(opps[["is_closed", "is_successful"]].value_counts().to_string())
    print(f"\n💰 Deal amount distribution:")
    print(opps["deal_amount"].describe().to_string())
    print(f"\n📈 Win rate by gift type (closed opps):")
    closed = opps[opps["is_closed"]]
    print(closed.groupby("gift_type")["is_successful"].mean().to_string())
    print(f"\n📁 Saved:")
    print(f"   data/synthetic/synthetic_opportunities.csv")
    print(f"   data/synthetic/opportunity_stages.csv")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    generate_opportunities()