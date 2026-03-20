"""
Synthetic Data Generator for Customer Churn Prediction.

Generates a realistic dataset of 5,000 customers with correlated features
and ~20% churn rate.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer churn dataset with realistic correlations."""
    rng = np.random.default_rng(seed)

    # --- Base features ---
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]

    # Signup dates: between 2021-01-01 and 2025-06-30
    signup_start = pd.Timestamp("2021-01-01")
    signup_range_days = (pd.Timestamp("2025-06-30") - signup_start).days
    signup_offsets = rng.integers(0, signup_range_days, size=n_customers)
    signup_dates = [signup_start + pd.Timedelta(days=int(d)) for d in signup_offsets]

    # Plan type distribution
    plan_types = rng.choice(
        ["Basic", "Standard", "Premium"],
        size=n_customers,
        p=[0.45, 0.35, 0.20],
    )

    # Total orders: depends on plan type
    plan_order_map = {"Basic": (5, 15), "Standard": (10, 40), "Premium": (20, 80)}
    total_orders = np.array([
        rng.integers(plan_order_map[p][0], plan_order_map[p][1] + 1)
        for p in plan_types
    ])

    # Avg order value: correlated with plan
    plan_aov_map = {"Basic": (15, 8), "Standard": (35, 12), "Premium": (70, 20)}
    avg_order_value = np.array([
        max(5, rng.normal(plan_aov_map[p][0], plan_aov_map[p][1]))
        for p in plan_types
    ]).round(2)

    # Support tickets: Poisson distributed, higher for Basic
    plan_ticket_map = {"Basic": 2.5, "Standard": 1.5, "Premium": 0.8}
    support_tickets = np.array([
        rng.poisson(plan_ticket_map[p]) for p in plan_types
    ])

    # Email open rate: Beta distributed
    email_open_rate = rng.beta(3, 5, size=n_customers).round(4)

    # Days since last login
    days_since_last_login = np.clip(
        rng.exponential(scale=20, size=n_customers).astype(int), 0, 180
    )

    # Discount used percentage
    discount_used_pct = rng.beta(2, 5, size=n_customers).round(4)

    # Reference date for recency calculations
    reference_date = pd.Timestamp("2025-12-01")

    # Last purchase date: between signup and reference date
    tenure_days = np.array([(reference_date - s).days for s in signup_dates])
    recency_raw = rng.exponential(scale=30, size=n_customers).astype(int)
    recency_raw = np.clip(recency_raw, 1, tenure_days)
    last_purchase_dates = [
        reference_date - pd.Timedelta(days=int(d)) for d in recency_raw
    ]

    # --- Generate churn with realistic correlations ---
    # Higher churn probability when:
    #   - More support tickets
    #   - Longer since last login
    #   - Lower email open rate
    #   - Basic plan
    #   - Fewer total orders
    #   - Higher recency (days since last purchase)
    plan_churn_bias = np.array(
        [0.6 if p == "Basic" else 0.0 if p == "Standard" else -0.8 for p in plan_types]
    )

    logit = (
        -2.0
        + 0.25 * (support_tickets - 2)
        + 0.03 * (days_since_last_login - 20)
        - 3.0 * (email_open_rate - 0.35)
        - 0.02 * (total_orders - 20)
        + 0.01 * (recency_raw - 30)
        + plan_churn_bias
        - 1.5 * (discount_used_pct - 0.25)
    )

    churn_prob = 1 / (1 + np.exp(-logit))
    churned = (rng.random(n_customers) < churn_prob).astype(int)

    # Adjust to ~20% churn rate by shifting the intercept if needed
    actual_rate = churned.mean()
    if abs(actual_rate - 0.20) > 0.02:
        # Re-calibrate
        from scipy.special import logit as sp_logit, expit
        shift = sp_logit(0.20) - sp_logit(actual_rate)
        adjusted_prob = expit(np.log(churn_prob / (1 - churn_prob)) + shift)
        churned = (rng.random(n_customers) < adjusted_prob).astype(int)

    # Introduce ~2% missing values in select columns for realism
    for col_data in [email_open_rate, days_since_last_login, discount_used_pct]:
        mask = rng.random(n_customers) < 0.02
        col_data[mask] = np.nan if col_data.dtype == float else -1

    # For integer column, use -1 as sentinel, we'll convert to NaN in DataFrame
    days_since_last_login = days_since_last_login.astype(float)
    days_since_last_login[days_since_last_login == -1] = np.nan

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates,
        "last_purchase_date": last_purchase_dates,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "support_tickets": support_tickets,
        "email_open_rate": email_open_rate,
        "days_since_last_login": days_since_last_login,
        "plan_type": plan_types,
        "discount_used_pct": discount_used_pct,
        "churned": churned,
    })

    return df


def main():
    """Generate and save synthetic dataset."""
    output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data()

    output_path = output_dir / "customers.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} rows → {output_path}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(f"Plan distribution:\n{df['plan_type'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


if __name__ == "__main__":
    main()
