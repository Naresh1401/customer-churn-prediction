"""
Feature engineering module for Customer Churn Prediction.

Creates derived features, encodes categoricals, and scales numerics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


REFERENCE_DATE = pd.Timestamp("2025-12-01")

NUMERIC_FEATURES = [
    "total_orders",
    "avg_order_value",
    "support_tickets",
    "email_open_rate",
    "days_since_last_login",
    "discount_used_pct",
    "recency",
    "frequency",
    "monetary",
    "engagement_score",
]


def compute_recency(df: pd.DataFrame, reference_date: pd.Timestamp = REFERENCE_DATE) -> pd.Series:
    """Days since last purchase relative to a reference date."""
    last_purchase = pd.to_datetime(df["last_purchase_date"])
    recency = (reference_date - last_purchase).dt.days
    return recency.astype(float)


def compute_frequency(df: pd.DataFrame) -> pd.Series:
    """Proxy for orders in last 90 days — scaled from total_orders and recency."""
    recency = compute_recency(df)
    # Heuristic: if customer purchased recently, higher proportion of orders are recent
    recency_factor = np.clip(1 - (recency / 365), 0.1, 1.0)
    frequency = (df["total_orders"] * recency_factor).round(0).astype(float)
    return frequency


def compute_monetary(df: pd.DataFrame) -> pd.Series:
    """Lifetime monetary value: avg_order_value × total_orders."""
    return (df["avg_order_value"] * df["total_orders"]).astype(float)


def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """Weighted engagement: 0.6 × email_open_rate + 0.4 × login_recency_score."""
    login_recency_score = 1 - np.clip(df["days_since_last_login"] / 180, 0, 1)
    score = 0.6 * df["email_open_rate"] + 0.4 * login_recency_score
    return score.round(4).astype(float)


def compute_risk_flag(df: pd.DataFrame) -> pd.Series:
    """1 if support_tickets > 3 AND days_since_last_login > 30."""
    flag = ((df["support_tickets"] > 3) & (df["days_since_last_login"] > 30)).astype(int)
    return flag


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features to the DataFrame."""
    df = df.copy()
    df["recency"] = compute_recency(df)
    df["frequency"] = compute_frequency(df)
    df["monetary"] = compute_monetary(df)
    df["engagement_score"] = compute_engagement_score(df)
    df["risk_flag"] = compute_risk_flag(df)
    return df


def encode_plan_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode plan_type, dropping the first category to avoid collinearity."""
    df = df.copy()
    dummies = pd.get_dummies(df["plan_type"], prefix="plan", drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["plan_type"])
    return df


def scale_features(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler on numeric features. Returns (df, fitted_scaler)."""
    if numeric_cols is None:
        numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler


def prepare_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Full feature pipeline: engineer → encode → drop dates → scale."""
    df = add_engineered_features(df)
    df = encode_plan_type(df)

    # Drop raw date columns (not needed for modelling)
    date_cols = [c for c in ["signup_date", "last_purchase_date"] if c in df.columns]
    df = df.drop(columns=date_cols)

    df, scaler = scale_features(df, scaler=scaler, fit=fit)
    return df, scaler
