"""
FastAPI application for Customer Churn Prediction.

Exposes a POST /predict endpoint that returns churn probability,
risk tier, and SHAP-based plain-English explanations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict_single

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predict customer churn probability with explainable AI.",
)

# Load model once at startup
_model, _scaler, _feature_names = None, None, None


@app.on_event("startup")
def startup_load_model():
    global _model, _scaler, _feature_names
    _model, _scaler, _feature_names = load_model()


# --------------- Request / Response schemas ---------------


class PredictRequest(BaseModel):
    customer_id: str
    features: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customer_id": "CUST-00042",
                    "features": {
                        "signup_date": "2022-03-15",
                        "last_purchase_date": "2025-10-01",
                        "total_orders": 12,
                        "avg_order_value": 45.50,
                        "support_tickets": 5,
                        "email_open_rate": 0.15,
                        "days_since_last_login": 45,
                        "plan_type": "Basic",
                        "discount_used_pct": 0.10,
                    },
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    top_reasons: list[str]


# --------------- Endpoints ---------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict churn probability for a single customer."""
    try:
        result = predict_single(
            features=request.features,
            model=_model,
            scaler=_scaler,
            feature_names=_feature_names,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        customer_id=request.customer_id,
        churn_probability=result["churn_probability"],
        risk_tier=result["risk_tier"],
        top_reasons=result["top_reasons"],
    )
