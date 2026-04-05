import uuid
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta


# -----------------------------
# HELPERS
# -----------------------------
def random_date(start_year=2019, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_id(prefix):
    # UUID guarantees uniqueness — no collisions
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def assign_gift_type(amount):
    """
    Classify gift type based on deal amount.
    Thresholds calibrated to the synthetic deal_amount distribution.
    """
    if amount < 5_000:
        return "Annual Giving"
    elif amount < 25_000:
        return "Major Gifts"
    else:
        return "Planned Giving"


def synthetic_ask_amount(avg_gift: float) -> int:
    """
    Generate a realistic pipeline ask amount with enough spread
    to produce all three gift type tiers.
    Draws from three tiers directly so distribution is controlled.
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


def stage_probability(stage):
    mapping = {
        "Discovery": random.randint(5, 20),
        "Engagement": random.randint(15, 35),
        "Evaluation": random.randint(30, 55),
        "Proposal": random.randint(50, 75),
        "Negotiation": random.randint(70, 90),
        "Committed": 100,
        "Lost": 0,
        "Rejected": 0,
    }
    return mapping[stage]


# Stage progression order — a gift can only move forward
STAGE_PROGRESSION = [
    "Discovery",
    "Engagement",
    "Evaluation",
    "Proposal",
    "Negotiation",
    "Committed",
]

TERMINAL_STAGES = ["Committed", "Lost", "Rejected"]


def simulate_lifecycle(start_date: datetime) -> list[dict]:
    """
    Simulate a gift's journey through pipeline stages.

    Returns a list of stage transition dicts, each representing one row
    in the opportunity_stages table. The gift either:
    - Progresses through some stages and closes (Committed)
    - Stalls and is lost (Lost / Rejected)
    - Is still open (stops mid-progression, no terminal stage)

    Each transition has a date so the full lifecycle is reconstructable.
    """
    transitions = []
    current_date = start_date

    # Pick a random entry stage (most gifts start at Discovery, some later)
    entry_idx = random.choices(
        range(len(STAGE_PROGRESSION)),
        weights=[40, 25, 15, 10, 7, 3],
        k=1,
    )[0]

    current_stage = STAGE_PROGRESSION[entry_idx]
    previous_stage = None

    while True:
        transitions.append(
            {
                "stage": current_stage,
                "previous_stage": previous_stage,
                "stage_entry_date": current_date.strftime("%Y-%m-%d"),
                "stage_probability_pct": stage_probability(current_stage),
                "is_current": False,  # will set last one to True below
            }
        )

        # Advance date by a realistic number of days per stage
        days_in_stage = random.randint(14, 120)
        current_date = current_date + timedelta(days=days_in_stage)

        # Decide what happens next
        stage_idx = STAGE_PROGRESSION.index(current_stage)
        is_last_stage = stage_idx == len(STAGE_PROGRESSION) - 1

        if is_last_stage:
            # Already at Committed — no further transition needed
            break

        # Roll for progression vs stall vs open
        roll = random.random()

        if roll < 0.55:
            # Advance to next stage
            previous_stage = current_stage
            current_stage = STAGE_PROGRESSION[stage_idx + 1]

        elif roll < 0.70:
            # Lost or Rejected
            terminal = random.choice(["Lost", "Rejected"])
            transitions.append(
                {
                    "stage": terminal,
                    "previous_stage": current_stage,
                    "stage_entry_date": current_date.strftime("%Y-%m-%d"),
                    "stage_probability_pct": 0,
                    "is_current": False,
                }
            )
            break

        else:
            # Still open — stop here, no terminal stage
            break

    # Mark the last transition as the current stage
    if transitions:
        transitions[-1]["is_current"] = True

    return transitions


def is_closed_won(transitions: list[dict]) -> tuple[bool, bool]:
    """Derive is_closed and is_successful from the terminal stage."""
    last_stage = transitions[-1]["stage"]
    if last_stage == "Committed":
        return True, True
    elif last_stage in ["Lost", "Rejected"]:
        return True, False
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
    gifts = pd.read_csv("data/raw/donation_records.csv")

    # -----------------------------
    # DETECT CAPACITY COLUMN
    # -----------------------------
    capacity_col = None
    for col in donors.columns:
        if "capacity" in col.lower() or "wealth" in col.lower():
            capacity_col = col
            break

    capacity_options = [
        "50K-100K",
        "100K-250K",
        "250K-500K",
        "500K-1M",
        "1M-2M",
        "2M-5M",
        "5M+",
    ]

    # -----------------------------
    # DERIVE DONOR FEATURES
    # -----------------------------
    if "donor_id" in gifts.columns:
        numeric_cols = gifts.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "donor_id"]
        preferred = [
            c for c in numeric_cols if "amount" in c.lower() or "value" in c.lower()
        ]
        amount_col = (
            preferred[0] if preferred else (numeric_cols[0] if numeric_cols else None)
        )

        if amount_col:
            print(f"💰 Using '{amount_col}' as donation column")
            donor_stats = (
                gifts.groupby("donor_id")
                .agg(avg_gift=(amount_col, "mean"), gift_count=(amount_col, "count"))
                .reset_index()
            )
            donors = donors.merge(donor_stats, on="donor_id", how="left")

    donors["avg_gift"] = donors.get("avg_gift", pd.Series()).fillna(500)
    donors["gift_count"] = donors.get("gift_count", pd.Series()).fillna(1)

    # -----------------------------
    # STABLE CONTACT ID MAP
    # 1:1 between donor_id and contact_id
    # -----------------------------
    contact_id_map = {
        donor_id: generate_id("contact") for donor_id in donors["donor_id"].unique()
    }

    # -----------------------------
    # GENERATE OPPORTUNITIES + STAGE HISTORY
    # -----------------------------
    AVG_OPPS_PER_DONOR = 3.5
    MAX_OPPS_PER_DONOR = 10

    opp_rows = []  # one row per opportunity (current state)
    stage_rows = []  # one row per stage transition (full history)

    for _, donor in donors.iterrows():

        donor_id = donor["donor_id"]
        contact_id = contact_id_map[donor_id]

        avg_gift = donor.get("avg_gift", 500)
        if pd.isna(avg_gift) or avg_gift <= 0:
            avg_gift = 500

        num_opps = min(np.random.poisson(AVG_OPPS_PER_DONOR), MAX_OPPS_PER_DONOR)
        num_opps = max(num_opps, 1)

        for _ in range(num_opps):

            opp_id = generate_id("opp")
            start_date = random_date()
            amount = synthetic_ask_amount(avg_gift)
            gift_type = assign_gift_type(amount)

            fiscal_year = start_date.year + (1 if start_date.month >= 7 else 0)
            fiscal_quarter = ((start_date.month - 1) // 3) + 1

            capacity_value = (
                donor.get(capacity_col)
                if capacity_col and pd.notna(donor.get(capacity_col))
                else random.choice(capacity_options)
            )

            # Simulate full lifecycle
            transitions = simulate_lifecycle(start_date)
            is_closed, is_successful = is_closed_won(transitions)

            current = transitions[-1]  # latest stage = current state

            # ── opportunities table (one row, current state) ──────────────────
            opp_rows.append(
                {
                    "opportunity_id": opp_id,
                    "donor_id": donor_id,
                    "contact_id": contact_id,
                    "sector": donor.get("sector", "Unknown"),
                    "capacity_band": capacity_value,
                    "gift_type": gift_type,
                    "deal_amount": amount,
                    "current_stage": current["stage"],
                    "current_stage_probability_pct": current["stage_probability_pct"],
                    "stage_entry_date": current["stage_entry_date"],
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "is_closed": is_closed,
                    "is_successful": is_successful,
                    "total_stage_count": len(transitions),
                }
            )

            # ── opportunity_stages table (one row per transition) ─────────────
            for t in transitions:
                stage_rows.append(
                    {
                        "opportunity_id": opp_id,
                        "donor_id": donor_id,
                        "stage": t["stage"],
                        "previous_stage": t["previous_stage"],
                        "stage_entry_date": t["stage_entry_date"],
                        "stage_probability_pct": t["stage_probability_pct"],
                        "is_current": t["is_current"],
                        "deal_amount": amount,
                        "gift_type": gift_type,
                    }
                )

    opps = pd.DataFrame(opp_rows)
    stages = pd.DataFrame(stage_rows)

    # -----------------------------
    # SAVE
    # -----------------------------
    os.makedirs("data/synthetic", exist_ok=True)

    opps.to_csv("data/synthetic/synthetic_opportunities.csv", index=False)
    stages.to_csv("data/synthetic/opportunity_stages.csv", index=False)

    print(f"\n✅ Opportunities generated:   {len(opps):,}")
    print(f"📋 Stage transitions logged:  {len(stages):,}")
    print(f"📊 Avg opps per donor:        {len(opps) / len(donors):.1f}")
    print(f"📊 Avg stages per opp:        {len(stages) / len(opps):.1f}")
    print(f"\n🎁 Gift type distribution:")
    print(opps["gift_type"].value_counts().to_string())
    print(f"\n🏁 Outcome distribution:")
    print(opps[["is_closed", "is_successful"]].value_counts().to_string())
    print(f"\n💰 Deal amount stats:")
    print(opps["deal_amount"].describe().apply(lambda x: f"${x:,.0f}").to_string())
    print(f"\n📁 Saved:")
    print(f"   data/synthetic/synthetic_opportunities.csv")
    print(f"   data/synthetic/opportunity_stages.csv")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    generate_opportunities()
