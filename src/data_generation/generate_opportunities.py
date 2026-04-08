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


def parse_capacity(band: str) -> tuple[float, float]:
    """
    Parse capacity band string into (min, max) numeric values.
    Mirrors the parse_capacity logic from the McGill notebook.
    """
    mapping = {
        "50K-100K":   (50_000,   100_000),
        "100K-250K":  (100_000,  250_000),
        "250K-500K":  (250_000,  500_000),
        "500K-1M":    (500_000,  1_000_000),
        "1M-2M":      (1_000_000, 2_000_000),
        "2M-5M":      (2_000_000, 5_000_000),
        "5M+":        (5_000_000, np.nan),
    }
    return mapping.get(band, (np.nan, np.nan))


def assign_gift_type(amount: int) -> str:
    if amount < 5_000:
        return "Annual Giving"
    elif amount < 25_000:
        return "Major Gifts"
    else:
        return "Planned Giving"


def synthetic_ask_amount() -> int:
    """Draw from three tiers to ensure meaningful gift type distribution."""
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


def base_win_prob(gift_type: str, amount: float, relationship_score: float) -> float:
    """
    Win probability driven by gift type, deal size, and relationship score.
    Mirrors the McGill model's use of Amount, capacity, and implicit relationship
    quality signal (fundraiser Probability field).

    relationship_score (1-10) is the strongest single predictor — fundraisers
    who rate a relationship highly are right more often than not.
    """
    # Base rate by gift type
    base = {
        "Annual Giving":  0.55,
        "Major Gifts":    0.38,
        "Planned Giving": 0.22,
    }.get(gift_type, 0.38)

    # Deal size modifier — larger deals are harder to close
    if amount > 200_000:
        base *= 0.60
    elif amount > 50_000:
        base *= 0.80
    elif amount < 5_000:
        base *= 1.20

    # Relationship score modifier — scale linearly from -0.15 to +0.15
    # Score 5 = neutral, score 10 = +0.15, score 1 = -0.15
    rel_modifier = (relationship_score - 5.5) / 5.5 * 0.20
    prob = base + rel_modifier

    return float(np.clip(prob, 0.05, 0.95))


def generate_relationship_score(win_prob_base: float) -> float:
    """
    Generate a relationship score (1-10) correlated with win probability.
    High-win opportunities tend to have been better qualified by the fundraiser.
    Adds noise so it's not perfectly correlated.
    """
    # Center around win_prob * 10, add noise
    center = win_prob_base * 10
    score  = np.random.normal(loc=center, scale=1.8)
    return round(float(np.clip(score, 1.0, 10.0)), 1)


STAGE_PROGRESSION = [
    "Discovery", "Engagement", "Evaluation",
    "Proposal", "Negotiation", "Committed",
]


def simulate_lifecycle(start_date: datetime, win_prob: float) -> list[dict]:
    """
    Simulate stage progression. win_prob influences advancement rate.
    Returns list of stage transition dicts for opportunity_stages table.
    """
    transitions = []
    current_date = start_date

    entry_weights = [40, 25, 15, 10, 7, 3]
    if win_prob > 0.5:
        entry_weights = [20, 20, 20, 20, 15, 5]

    entry_idx      = random.choices(range(len(STAGE_PROGRESSION)), weights=entry_weights, k=1)[0]
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

        stage_idx     = STAGE_PROGRESSION.index(current_stage)
        is_last_stage = (stage_idx == len(STAGE_PROGRESSION) - 1)

        if is_last_stage:
            break

        advance_prob = 0.45 + (win_prob * 0.25)
        stall_prob   = advance_prob + 0.15
        roll         = random.random()

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
            break

    if transitions:
        transitions[-1]["is_current"] = True

    return transitions


def is_closed_won(transitions: list[dict], win_prob: float) -> tuple[bool, bool]:
    last_stage = transitions[-1]["stage"]
    if last_stage == "Committed":
        return True, random.random() < 0.88
    elif last_stage in ["Lost", "Rejected"]:
        return True, random.random() < 0.04
    else:
        return False, False


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_opportunities():

    print("🔄 Starting generation...")

    donors = pd.read_csv("data/raw/donor_profiles.csv")
    gifts  = pd.read_csv("data/raw/donation_records.csv")

    capacity_options = [
        "50K-100K", "100K-250K", "250K-500K",
        "500K-1M",  "1M-2M",    "2M-5M",    "5M+",
    ]

    capacity_col = None
    for col in donors.columns:
        if "capacity" in col.lower() or "wealth" in col.lower():
            capacity_col = col
            break

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

    contact_id_map = {
        donor_id: generate_id("contact")
        for donor_id in donors["donor_id"].unique()
    }

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
            amount_log = float(np.log1p(amount))

            # Capacity band — use donor's if available, else random
            capacity_band = (
                donor.get(capacity_col)
                if capacity_col and pd.notna(donor.get(capacity_col))
                else random.choice(capacity_options)
            )
            capacity_min, capacity_max = parse_capacity(capacity_band)

            # Sector modifier on win probability
            sector_boost = {
                "Healthcare":  0.05,
                "Technology":  0.03,
                "Government": -0.05,
                "Education":   0.02,
            }.get(sector, 0.0)

            # Base win prob before relationship score
            base_prob = float(np.clip(
                {
                    "Annual Giving":  0.55,
                    "Major Gifts":    0.38,
                    "Planned Giving": 0.22,
                }.get(gift_type, 0.38)
                + sector_boost
                + (0.15 if amount < 5_000 else -0.15 if amount > 200_000 else 0),
                0.05, 0.95
            ))

            # Relationship score — correlated with base_prob, adds noise
            relationship_score = generate_relationship_score(base_prob)

            # Final win prob incorporating relationship score
            win_prob = base_win_prob(gift_type, amount, relationship_score)

            fiscal_year    = start_date.year + (1 if start_date.month >= 7 else 0)
            fiscal_quarter = ((start_date.month - 1) // 3) + 1

            transitions    = simulate_lifecycle(start_date, win_prob)
            is_closed, is_successful = is_closed_won(transitions, win_prob)

            current        = transitions[-1]

            # days_in_stage: days since current stage entry (mirrors McGill's days_in_stage)
            try:
                entry_dt    = datetime.strptime(current["stage_entry_date"], "%Y-%m-%d")
                days_in_stage = (datetime.today() - entry_dt).days
            except Exception:
                days_in_stage = 0

            opp_rows.append({
                "opportunity_id":                opp_id,
                "donor_id":                      donor_id,
                "contact_id":                    contact_id,
                "sector":                        sector,
                "capacity_band":                 capacity_band,
                "capacity_min":                  capacity_min,
                "capacity_max":                  capacity_max,
                "gift_type":                     gift_type,
                "deal_amount":                   amount,
                "amount_log":                    round(amount_log, 4),
                "relationship_score":            relationship_score,
                "current_stage":                 current["stage"],
                "current_stage_probability_pct": current["stage_probability_pct"],
                "stage_entry_date":              current["stage_entry_date"],
                "days_in_stage":                 days_in_stage,
                "fiscal_year":                   fiscal_year,
                "fiscal_quarter":                fiscal_quarter,
                "is_closed":                     is_closed,
                "is_successful":                 is_successful,
                "total_stage_count":             len(transitions),
                "win_prob_signal":               round(win_prob, 4),
            })

            for t in transitions:
                try:
                    t_entry   = datetime.strptime(t["stage_entry_date"], "%Y-%m-%d")
                    t_days    = (datetime.today() - t_entry).days
                except Exception:
                    t_days = 0

                stage_rows.append({
                    "opportunity_id":        opp_id,
                    "donor_id":              donor_id,
                    "stage":                 t["stage"],
                    "previous_stage":        t["previous_stage"],
                    "stage_entry_date":      t["stage_entry_date"],
                    "stage_probability_pct": t["stage_probability_pct"],
                    "is_current":            t["is_current"],
                    "deal_amount":           amount,
                    "amount_log":            round(amount_log, 4),
                    "gift_type":             gift_type,
                    "sector":                sector,
                    "capacity_band":         capacity_band,
                    "capacity_min":          capacity_min,
                    "capacity_max":          capacity_max,
                    "relationship_score":    relationship_score,
                    "days_in_stage":         t_days,
                    "fiscal_quarter":        fiscal_quarter,
                    "win_prob_signal":       round(win_prob, 4),
                })

    opps   = pd.DataFrame(opp_rows)
    stages = pd.DataFrame(stage_rows)

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
    print(f"\n📈 Win rate by gift type (closed opps):")
    closed = opps[opps["is_closed"]]
    print(closed.groupby("gift_type")["is_successful"].mean().round(3).to_string())
    print(f"\n📊 Relationship score stats:")
    print(opps["relationship_score"].describe().round(2).to_string())
    print(f"\n📁 Saved:")
    print(f"   data/synthetic/synthetic_opportunities.csv")
    print(f"   data/synthetic/opportunity_stages.csv")


if __name__ == "__main__":
    generate_opportunities()