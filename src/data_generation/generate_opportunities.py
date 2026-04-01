import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta


# -----------------------------
# HELPERS
# -----------------------------
def random_date(start_year=2019, end_year=2027):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_id(prefix):
    return f"{prefix}_{random.randint(1000000, 9999999)}"


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


def assign_outcome(stage, close_date, late_stages):
    today = datetime.today()

    if close_date > today:
        return False, False

    if stage in ["Lost", "Rejected"]:
        return True, False

    if stage in late_stages:
        return True, random.random() < 0.85

    return True, random.random() < 0.4


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

    if capacity_col:
        print(f"📊 Using '{capacity_col}' as capacity column")
    else:
        print("⚠️ No capacity column found — generating synthetic capacity")

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
    # CONFIG
    # -----------------------------
    AVG_OPPS_PER_DONOR = 3.5
    MAX_OPPS_PER_DONOR = 10

    pipeline_stages = [
        "Discovery",
        "Engagement",
        "Evaluation",
        "Proposal",
        "Negotiation",
        "Committed",
        "Lost",
        "Rejected",
    ]

    late_stages = ["Negotiation", "Committed"]

    # -----------------------------
    # DERIVE DONOR FEATURES (FIXED)
    # -----------------------------
    if "donor_id" in gifts.columns:

        numeric_cols = gifts.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "donor_id"]

        preferred_cols = [
            c for c in numeric_cols if "amount" in c.lower() or "value" in c.lower()
        ]

        if preferred_cols:
            amount_col = preferred_cols[0]
        elif numeric_cols:
            amount_col = numeric_cols[0]
        else:
            amount_col = None

        if amount_col:
            print(f"💰 Using '{amount_col}' as donation column")

            donor_stats = (
                gifts.groupby("donor_id")
                .agg(
                    avg_gift=(amount_col, "mean"),
                    gift_count=(amount_col, "count"),
                )
                .reset_index()
            )

            donors = donors.merge(donor_stats, on="donor_id", how="left")
        else:
            print("⚠️ No usable numeric column found")

    donors["avg_gift"] = donors.get("avg_gift", pd.Series()).fillna(10000)
    donors["gift_count"] = donors.get("gift_count", pd.Series()).fillna(1)

    # -----------------------------
    # GENERATE OPPORTUNITIES
    # -----------------------------
    rows = []

    for _, donor in donors.iterrows():

        donor_id = donor["donor_id"]

        num_opps = min(
            np.random.poisson(AVG_OPPS_PER_DONOR),
            MAX_OPPS_PER_DONOR,
        )
        num_opps = max(num_opps, 1)

        for _ in range(num_opps):

            stage = random.choice(pipeline_stages)
            close_date = random_date()
            is_closed, is_won = assign_outcome(stage, close_date, late_stages)

            avg_gift = donor.get("avg_gift", 10000)

            # FIXED amount generation
            if pd.isna(avg_gift) or avg_gift <= 0:
                avg_gift = random.randint(5000, 25000)

            amount = int(np.random.lognormal(mean=np.log(avg_gift), sigma=0.75))

            amount = max(amount, 1000)
            amount = min(amount, avg_gift * 20)

            fiscal_year = close_date.year + (1 if close_date.month >= 7 else 0)
            fiscal_quarter = ((close_date.month - 1) // 3) + 1

            capacity_value = (
                donor.get(capacity_col)
                if capacity_col and pd.notna(donor.get(capacity_col))
                else random.choice(capacity_options)
            )

            row = {
                "sector": donor.get("sector", "Unknown"),
                "capacity_band": capacity_value,
                "stage": stage,
                "deal_amount": amount,
                "stage_probability_pct": stage_probability(stage),
                "expected_close_date": close_date.strftime("%Y-%m-%d"),
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "is_closed": is_closed,
                "is_successful": is_won,
                "opportunity_id": generate_id("opp"),
                "donor_id": donor_id,
                "contact_id": donor.get("contact_id", generate_id("contact")),
            }

            rows.append(row)

    opps = pd.DataFrame(rows)

    # -----------------------------
    # SAVE
    # -----------------------------
    os.makedirs("data/synthetic", exist_ok=True)

    output_path = "data/synthetic/synthetic_opportunities.csv"
    opps.to_csv(output_path, index=False)

    print(f"✅ Opportunities generated: {len(opps)}")
    print(f"📊 Avg opps per donor: {len(opps) / len(donors)}")
    print(f"📁 Saved to: {output_path}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    generate_opportunities()
