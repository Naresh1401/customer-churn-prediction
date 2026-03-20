"""
Unit tests for the Customer Churn Prediction project.

Tests cover feature engineering, prediction outputs, and API responses.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.feature_engineering import (
    add_engineered_features,
    compute_engagement_score,
    compute_frequency,
    compute_monetary,
    compute_recency,
    compute_risk_flag,
    prepare_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """A small sample DataFrame mimicking the raw dataset."""
    return pd.DataFrame(
        {
            "customer_id": ["CUST-00001", "CUST-00002", "CUST-00003"],
            "signup_date": pd.to_datetime(
                ["2022-01-01", "2023-06-15", "2024-03-01"]
            ),
            "last_purchase_date": pd.to_datetime(
                ["2025-11-01", "2025-09-15", "2025-06-01"]
            ),
            "total_orders": [50, 10, 5],
            "avg_order_value": [75.0, 30.0, 15.0],
            "support_tickets": [1, 5, 4],
            "email_open_rate": [0.55, 0.10, 0.20],
            "days_since_last_login": [3, 45, 60],
            "plan_type": ["Premium", "Basic", "Standard"],
            "discount_used_pct": [0.30, 0.05, 0.12],
            "churned": [0, 1, 1],
        }
    )


@pytest.fixture
def sample_features():
    """A feature dict suitable for the /predict endpoint."""
    return {
        "signup_date": "2022-01-01",
        "last_purchase_date": "2025-11-01",
        "total_orders": 50,
        "avg_order_value": 75.0,
        "support_tickets": 1,
        "email_open_rate": 0.55,
        "days_since_last_login": 3,
        "plan_type": "Premium",
        "discount_used_pct": 0.30,
    }


# ---------------------------------------------------------------------------
# Feature engineering — dtype checks
# ---------------------------------------------------------------------------


class TestFeatureEngineeringDtypes:
    def test_recency_dtype(self, sample_df):
        result = compute_recency(sample_df)
        assert result.dtype == np.float64

    def test_frequency_dtype(self, sample_df):
        result = compute_frequency(sample_df)
        assert result.dtype == np.float64

    def test_monetary_dtype(self, sample_df):
        result = compute_monetary(sample_df)
        assert result.dtype == np.float64

    def test_engagement_score_dtype(self, sample_df):
        result = compute_engagement_score(sample_df)
        assert result.dtype == np.float64

    def test_risk_flag_dtype(self, sample_df):
        result = compute_risk_flag(sample_df)
        assert result.dtype in (np.int64, np.int32, int)

    def test_add_engineered_features_columns(self, sample_df):
        result = add_engineered_features(sample_df)
        expected_cols = {"recency", "frequency", "monetary", "engagement_score", "risk_flag"}
        assert expected_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Recency calculation accuracy
# ---------------------------------------------------------------------------


class TestRecencyAccuracy:
    def test_recency_values(self, sample_df):
        """Recency should match hand-calculated days from reference date."""
        from src.feature_engineering import REFERENCE_DATE

        recency = compute_recency(sample_df)
        expected = [
            (REFERENCE_DATE - pd.Timestamp("2025-11-01")).days,
            (REFERENCE_DATE - pd.Timestamp("2025-09-15")).days,
            (REFERENCE_DATE - pd.Timestamp("2025-06-01")).days,
        ]
        assert list(recency) == expected

    def test_recency_non_negative(self, sample_df):
        recency = compute_recency(sample_df)
        assert (recency >= 0).all()


# ---------------------------------------------------------------------------
# Risk flag logic
# ---------------------------------------------------------------------------


class TestRiskFlag:
    def test_risk_flag_values(self, sample_df):
        flags = compute_risk_flag(sample_df)
        # CUST-00001: tickets=1, login=3 → 0
        # CUST-00002: tickets=5, login=45 → 1
        # CUST-00003: tickets=4, login=60 → 1
        assert list(flags) == [0, 1, 1]


# ---------------------------------------------------------------------------
# Model prediction range
# ---------------------------------------------------------------------------


class TestModelPrediction:
    """Tests that require a trained model on disk.
    Skipped gracefully if no model exists (e.g., in CI before training)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        from pathlib import Path
        model_path = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"
        if not model_path.exists():
            pytest.skip("Trained model not found — run training first.")

    def test_predict_returns_valid_probability(self, sample_features):
        from src.predict import predict_single

        result = predict_single(sample_features)
        prob = result["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_returns_risk_tier(self, sample_features):
        from src.predict import predict_single

        result = predict_single(sample_features)
        assert result["risk_tier"] in {"Low", "Medium", "High"}

    def test_predict_returns_reasons(self, sample_features):
        from src.predict import predict_single

        result = predict_single(sample_features)
        assert isinstance(result["top_reasons"], list)
        assert len(result["top_reasons"]) > 0


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestAPI:
    """Tests for the FastAPI /predict endpoint.
    Skipped gracefully if no model exists."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        from pathlib import Path
        model_path = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"
        if not model_path.exists():
            pytest.skip("Trained model not found — run training first.")

    @pytest.fixture
    def client(self):
        from api.main import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_200(self, client, sample_features):
        payload = {"customer_id": "CUST-TEST", "features": sample_features}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

        body = resp.json()
        assert "churn_probability" in body
        assert "risk_tier" in body
        assert "top_reasons" in body
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_predict_invalid_payload(self, client):
        resp = client.post("/predict", json={"bad_key": "value"})
        assert resp.status_code == 422
