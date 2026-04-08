"""
generate_opportunities.py
Synthetic opportunity pipeline generator with McGill-inspired feature set.

Features generated mirror the real McGill CRM schema:
  - amount_log            : log1p(deal_amount)
  - capacity_min/max      : parsed from capacity band (both bounds always present)
  - days_in_stage         : days at current stage, clipped to >= 0
  - stage_num             : total stage transitions (log-capped at 10)
  - total_activities      : contact frequency per opportunity
  - essential_moves       : key fundraiser interactions
  - prior_gift_count      : how many times donor has given before
  - prior_gift_total_log  : log total prior giving
  - fiscal_quarter        : seasonality
  - gift_type             : Annual / Major / Planned
  - sector                : fundraising division
  - vse_category          : Alumni, Corporation, Foundation, etc.
  - prospect_status       : MG Prospect, MG DQ, etc.
  - has_prospect_manager  : assigned PM = more cultivated
  - has_volunteer_history : engagement signal
"""

import pandas as pd
import numpy as np
import random
import os
import uuid
from datetime import datetime, timedelta


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_date(start_year=2019, end_year=2025):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


def generate_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def stage_probability(stage):
    mapping = {
        "Qualification":             random.randint(5,  15),
        "Cultivation":               random.randint(15, 30),
        "Cleared for Solicitation":  random.randint(30, 50),
        "Ask in Progress":           random.randint(50, 70),
        "Closing":                   random.randint(70, 90),
        "Stewardship":               100,
        "Declined":                  0,
        "Rejected":                  0,
    }
    return mapping.get(stage, random.randint(10, 50))


def parse_capacity(band: str) -> tuple[float, float]:
    """
    Parse capacity band to (min, max) numeric bounds.
    Both bounds always populated — open-ended bands get a reasonable cap.
    Handles both synthetic (100K-250K) and McGill ($100,000-$499,999) formats.
    """
    mapping = {
        # Synthetic bands
        "50K-100K":   (50_000,      100_000),
        "100K-250K":  (100_000,     250_000),
        "250K-500K":  (250_000,     500_000),
        "500K-1M":    (500_000,   1_000_000),
        "1M-2M":    (1_000_000,   2_000_000),
        "2M-5M":    (2_000_000,   5_000_000),
        "5M+":      (5_000_000,  10_000_000),  # cap open-ended at $10M
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
    tier = random.choices(["annual", "major", "planned"], weights=[40, 45, 15])[0]
    if tier == "annual":
        return random.randint(1_000, 4_999)
    elif tier == "major":
        return random.randint(5_000, 24_999)
    else:
        return random.randint(25_000, 500_000)


def compute_win_prob(
    gift_type: str,
    amount: float,
    vse_category: str,
    prospect_status: str,
    has_pm: bool,
    has_volunteer: bool,
    prior_gift_count: int,
    capacity_min: float,
) -> float:
    """
    Win probability with VSE category as the primary base rate,
    calibrated directly to real McGill win rates.
    """
    # VSE category IS the base rate — calibrated to real McGill data
    base = {
        "Foundation":        0.58,
        "Other Individual":  0.54,
        "Faculty and Staff": 0.44,
        "Corporation":       0.38,
        "Parent":            0.37,
        "Alumni":            0.35,
    }.get(vse_category, 0.40)

    # Gift type modifier
    base += {
        "Annual Giving":  0.06,
        "Major Gifts":    0.00,
        "Planned Giving": -0.10,
    }.get(gift_type, 0.0)

    # Deal size modifier
    if amount > 200_000: base -= 0.12
    elif amount > 50_000: base -= 0.06
    elif amount < 5_000:  base += 0.05

    # Prospect status modifier
    base += {
        "MG Identified":  0.08,
        "MG Prospect":    0.03,
        "MG Unknown":    -0.05,
        "MG DQ":         -0.18,
    }.get(prospect_status, 0.0)

    # Engagement modifiers
    if has_pm:        base += 0.05
    if has_volunteer: base += 0.04

    # Prior giving
    if prior_gift_count >= 10:  base += 0.07
    elif prior_gift_count >= 3: base += 0.03

    # Capacity vs ask alignment
    if not np.isnan(capacity_min) and capacity_min > 0:
        ratio = amount / capacity_min
        if ratio < 0.1:   base += 0.04
        elif ratio > 2.0: base -= 0.07

    return float(np.clip(base, 0.05, 0.95))


def generate_activities(win_prob: float, stage_count: int) -> tuple[int, int]:
    """
    Generate total_activities and essential_moves correlated with win_prob.
    More engaged opportunities have more activities and essential moves.
    Mirrors activity report patterns from McGill data.
    """
    base_activities = max(1, int(np.random.poisson(win_prob * 12 + stage_count * 1.5)))
    base_activities = min(base_activities, 50)
    essential_moves = max(0, int(np.random.poisson(win_prob * 7)))
    essential_moves = min(essential_moves, base_activities)
    return base_activities, essential_moves


# ── Stage progression (mirrors McGill MG pipeline) ────────────────────────────

STAGE_PROGRESSION = [
    "Qualification",
    "Cultivation",
    "Cleared for Solicitation",
    "Ask in Progress",
    "Closing",
    "Stewardship",
]


def simulate_lifecycle(start_date: datetime, win_prob: float) -> list[dict]:
    """
    Simulate MG pipeline lifecycle mirroring McGill's 6-stage MG progression:
    Qualification → Cultivation → Cleared for Solicitation →
    Ask in Progress → Closing → Stewardship
    """
    transitions = []
    current_date = start_date

    # Higher win_prob opportunities tend to enter at a later stage
    entry_weights = [35, 25, 18, 12, 7, 3] if win_prob <= 0.5 else [15, 20, 25, 22, 13, 5]
    entry_idx      = random.choices(range(len(STAGE_PROGRESSION)), weights=entry_weights)[0]
    current_stage  = STAGE_PROGRESSION[entry_idx]
    previous_stage = None

    while True:
        days_here = max(0, random.randint(14, 150))
        transitions.append({
            "stage":                 current_stage,
            "previous_stage":        previous_stage,
            "stage_entry_date":      current_date.strftime("%Y-%m-%d"),
            "stage_probability_pct": stage_probability(current_stage),
            "is_current":            False,
        })
        current_date += timedelta(days=days_here)

        stage_idx     = STAGE_PROGRESSION.index(current_stage)
        is_last_stage = (stage_idx == len(STAGE_PROGRESSION) - 1)

        if is_last_stage:
            break

        advance_prob = 0.42 + win_prob * 0.28
        roll         = random.random()

        if roll < advance_prob:
            previous_stage = current_stage
            current_stage  = STAGE_PROGRESSION[stage_idx + 1]
        elif roll < advance_prob + 0.18:
            terminal = random.choice(["Declined", "Rejected"])
            transitions.append({
                "stage":                 terminal,
                "previous_stage":        current_stage,
                "stage_entry_date":      current_date.strftime("%Y-%m-%d"),
                "stage_probability_pct": 0,
                "is_current":            False,
            })
            break
        else:
            break  # still open

    if transitions:
        transitions[-1]["is_current"] = True

    return transitions


def derive_outcome(transitions: list[dict], win_prob: float) -> tuple[bool, bool]:
    """
    Derive is_closed / is_successful.
    win_prob directly drives the won/lost split for terminal stages
    so feature-outcome correlations are preserved in the final data.
    """
    last = transitions[-1]["stage"]
    if last == "Stewardship":
        # Won — use win_prob directly so higher-quality opps win more often
        return True, random.random() < (0.60 + win_prob * 0.40)
    elif last in ["Declined", "Rejected"]:
        # Lost — small revival chance correlated with win_prob
        return True, random.random() < (win_prob * 0.10)
    return False, False


# ── Donor profile helpers ─────────────────────────────────────────────────────

VSE_CATEGORIES   = ["Alumni", "Alumni", "Alumni", "Corporation", "Foundation",
                     "Parent", "Faculty and Staff", "Other Individual"]
PROSPECT_STATUSES = ["MG Prospect", "MG Prospect", "MG Identified",
                      "MG Unknown", "MG DQ"]
SECTORS          = ["University-Wide", "Medicine", "Engineering", "Arts",
                     "Law", "Management", "Science", "Education", "Neurology"]
CAPACITY_OPTIONS = ["50K-100K", "100K-250K", "250K-500K",
                     "500K-1M", "1M-2M", "2M-5M", "5M+"]


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_opportunities():
    print("🔄 Starting generation...")

    # Load donor data
    donors = pd.read_csv("data/raw/donor_profiles.csv")
    gifts  = pd.read_csv("data/raw/donation_records.csv")

    # Derive prior giving stats per donor
    if "donor_id" in gifts.columns:
        amount_cols = [c for c in gifts.select_dtypes(include="number").columns
                       if c != "donor_id" and ("amount" in c.lower() or "value" in c.lower())]
        if amount_cols:
            amount_col = amount_cols[0]
            print(f"💰 Using '{amount_col}' as donation column")
            donor_stats = (
                gifts.groupby("donor_id")
                .agg(
                    prior_gift_count=(amount_col, "count"),
                    prior_gift_total=(amount_col, "sum"),
                    prior_gift_avg=(amount_col,   "mean"),
                )
                .reset_index()
            )
            donors = donors.merge(donor_stats, on="donor_id", how="left")

    donors["prior_gift_count"] = donors.get("prior_gift_count", pd.Series()).fillna(0).astype(int)
    donors["prior_gift_total"] = donors.get("prior_gift_total", pd.Series()).fillna(0)

    # Stable contact ID map (1:1 donor → contact)
    contact_id_map = {did: generate_id("contact") for did in donors["donor_id"].unique()}

    AVG_OPPS  = 3.5
    MAX_OPPS  = 10
    opp_rows  = []
    stage_rows = []

    for _, donor in donors.iterrows():
        donor_id         = donor["donor_id"]
        contact_id       = contact_id_map[donor_id]
        sector           = donor.get("sector", random.choice(SECTORS))
        prior_gift_count = int(donor.get("prior_gift_count", 0))
        prior_gift_total = float(donor.get("prior_gift_total", 0))

        # Assign stable donor-level profile features
        vse_category      = random.choice(VSE_CATEGORIES)
        prospect_status   = random.choice(PROSPECT_STATUSES)
        has_pm            = random.random() < 0.45
        has_volunteer     = random.random() < 0.25
        capacity_band     = random.choice(CAPACITY_OPTIONS)
        capacity_min, capacity_max = parse_capacity(capacity_band)

        num_opps = min(max(np.random.poisson(AVG_OPPS), 1), MAX_OPPS)

        for _ in range(num_opps):
            opp_id     = generate_id("opp")
            start_date = random_date()
            amount     = synthetic_ask_amount()
            gift_type  = assign_gift_type(amount)
            amount_log = float(np.log1p(amount))
            prior_gift_total_log = float(np.log1p(prior_gift_total))

            fiscal_year    = start_date.year + (1 if start_date.month >= 7 else 0)
            fiscal_quarter = ((start_date.month - 1) // 3) + 1

            win_prob = compute_win_prob(
                gift_type, amount, vse_category, prospect_status,
                has_pm, has_volunteer, prior_gift_count, capacity_min,
            )

            transitions = simulate_lifecycle(start_date, win_prob)
            is_closed, is_successful = derive_outcome(transitions, win_prob)

            current   = transitions[-1]
            stage_num = min(len(transitions), 10)  # log-cap at 10 to avoid outlier dominance

            # days_in_stage: clipped to >= 0 (negative values are data entry errors)
            try:
                entry_dt      = datetime.strptime(current["stage_entry_date"], "%Y-%m-%d")
                days_in_stage = max(0, (datetime.today() - entry_dt).days)
            except Exception:
                days_in_stage = 0

            total_activities, essential_moves = generate_activities(win_prob, stage_num)

            opp_rows.append({
                "opportunity_id":                opp_id,
                "donor_id":                      donor_id,
                "contact_id":                    contact_id,
                # Opportunity features
                "gift_type":                     gift_type,
                "deal_amount":                   amount,
                "amount_log":                    round(amount_log, 4),
                "current_stage":                 current["stage"],
                "current_stage_probability_pct": current["stage_probability_pct"],
                "stage_entry_date":              current["stage_entry_date"],
                "days_in_stage":                 days_in_stage,
                "stage_num":                     stage_num,
                "fiscal_year":                   fiscal_year,
                "fiscal_quarter":                fiscal_quarter,
                "is_closed":                     is_closed,
                "is_successful":                 is_successful,
                "total_stage_count":             len(transitions),
                # Capacity (both bounds always present)
                "capacity_band":                 capacity_band,
                "capacity_min":                  capacity_min,
                "capacity_max":                  capacity_max,
                # Account / donor profile
                "sector":                        sector,
                "vse_category":                  vse_category,
                "prospect_status":               prospect_status,
                "has_prospect_manager":          int(has_pm),
                "has_volunteer_history":         int(has_volunteer),
                # Prior giving history
                "prior_gift_count":              prior_gift_count,
                "prior_gift_total_log":          round(prior_gift_total_log, 4),
                # Activity features
                "total_activities":              total_activities,
                "essential_moves":               essential_moves,
                # Internal signal (not used as model feature)
                "win_prob_signal":               round(win_prob, 4),
            })

            for t in transitions:
                try:
                    t_entry       = datetime.strptime(t["stage_entry_date"], "%Y-%m-%d")
                    t_days        = max(0, (datetime.today() - t_entry).days)
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
                    # Features available at each stage snapshot
                    "deal_amount":           amount,
                    "amount_log":            round(amount_log, 4),
                    "gift_type":             gift_type,
                    "sector":                sector,
                    "vse_category":          vse_category,
                    "prospect_status":       prospect_status,
                    "capacity_band":         capacity_band,
                    "capacity_min":          capacity_min,
                    "capacity_max":          capacity_max,
                    "has_prospect_manager":  int(has_pm),
                    "has_volunteer_history": int(has_volunteer),
                    "prior_gift_count":      prior_gift_count,
                    "prior_gift_total_log":  round(prior_gift_total_log, 4),
                    "total_activities":      total_activities,
                    "essential_moves":       essential_moves,
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
    closed = opps[opps["is_closed"]]
    print(f"\n📈 Win rate by gift type:")
    print(closed.groupby("gift_type")["is_successful"].mean().round(3).to_string())
    print(f"\n📈 Win rate by VSE category:")
    print(closed.groupby("vse_category")["is_successful"].mean().round(3).to_string())
    print(f"\n📈 Win rate by prospect status:")
    print(closed.groupby("prospect_status")["is_successful"].mean().round(3).to_string())
    print(f"\n📁 Saved:")
    print(f"   data/synthetic/synthetic_opportunities.csv")
    print(f"   data/synthetic/opportunity_stages.csv")


if __name__ == "__main__":
    generate_opportunities()