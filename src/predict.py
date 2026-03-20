"""
Prediction module for Customer Churn Prediction.

Loads the saved model artefact and provides a predict function used by
the FastAPI endpoint and batch scoring.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.feature_engineering import prepare_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def load_model(path: Path | str | None = None):
    """Load the persisted model artefact (model, scaler, feature_names)."""
    if path is None:
        path = MODEL_PATH
    artifact = joblib.load(path)
    return artifact["model"], artifact["scaler"], artifact["feature_names"]


def predict_single(features: dict, model=None, scaler=None, feature_names=None):
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    features : dict
        Raw feature values as they would appear in the CSV
        (before engineering / scaling).
    model, scaler, feature_names :
        Pre-loaded artefacts. If None they are loaded from disk.

    Returns
    -------
    dict with churn_probability, risk_tier, top_reasons
    """
    if model is None:
        model, scaler, feature_names = load_model()

    # Build a one-row DataFrame
    df = pd.DataFrame([features])

    # Ensure date columns are parsed
    for col in ["signup_date", "last_purchase_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Feature pipeline (use the same scaler)
    df_feat, _ = prepare_features(df, scaler=scaler, fit=False)

    # Align columns with training feature set
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    df_feat = df_feat[feature_names]

    # Predict probability
    prob = float(model.predict_proba(df_feat)[:, 1][0])

    # Risk tier
    if prob >= 0.7:
        risk_tier = "High"
    elif prob >= 0.4:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    # SHAP-based reasons
    top_reasons = _explain(model, df_feat, feature_names)

    return {
        "churn_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "top_reasons": top_reasons,
    }


def _explain(model, df_feat: pd.DataFrame, feature_names: list[str]) -> list[str]:
    """Return top-3 plain-English reasons from SHAP values."""
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, df_feat)
        sv = explainer.shap_values(df_feat)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv).flatten()
    except Exception:
        return ["Unable to compute explanations for this model type."]

    # Pair feature names with their SHAP impact
    impacts = sorted(
        zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True
    )

    reason_templates = {
        "recency": "Customer hasn't purchased in a long time",
        "support_tickets": "High number of support tickets",
        "days_since_last_login": "Customer hasn't logged in recently",
        "email_open_rate": "Low email engagement",
        "engagement_score": "Low overall engagement score",
        "risk_flag": "Customer flagged as at-risk",
        "frequency": "Low recent purchase frequency",
        "monetary": "Low lifetime monetary value",
        "total_orders": "Few total orders placed",
        "avg_order_value": "Low average order value",
        "discount_used_pct": "Low discount utilisation",
        "plan_Premium": "Not on a premium plan",
        "plan_Standard": "Not on a standard plan",
    }

    reasons = []
    for feat, val in impacts[:3]:
        direction = "increases" if val > 0 else "decreases"
        template = reason_templates.get(feat, f"{feat} value")
        reasons.append(f"{template} ({direction} churn risk)")

    return reasons
