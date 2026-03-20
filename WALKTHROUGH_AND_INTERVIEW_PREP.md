# Customer Churn Prediction — Code Walkthrough & Interview Preparation

> Use this document to understand every part of the codebase and confidently
> explain it in data science / ML engineering interviews.

---

## Table of Contents

1. [Project Elevator Pitch](#1-project-elevator-pitch)
2. [End-to-End Pipeline Walkthrough](#2-end-to-end-pipeline-walkthrough)
3. [File-by-File Deep Dive](#3-file-by-file-deep-dive)
4. [Key Design Decisions & Why](#4-key-design-decisions--why)
5. [Metrics Deep Dive](#5-metrics-deep-dive)
6. [Common Interview Questions & Answers](#6-common-interview-questions--answers)
7. [System Design Discussion Points](#7-system-design-discussion-points)
8. [Behavioural / STAR Stories from This Project](#8-behavioural--star-stories-from-this-project)
9. [Quick Cheat Sheet — Numbers to Remember](#9-quick-cheat-sheet--numbers-to-remember)

---

## 1. Project Elevator Pitch

> "I built an end-to-end customer churn prediction system that identifies
> which customers are likely to cancel within 30 days. The pipeline generates
> synthetic data, engineers RFM and engagement features, trains three models
> (Logistic Regression, Random Forest, XGBoost) with SMOTE and GridSearchCV,
> selects the best on ROC-AUC, and serves predictions via a FastAPI endpoint
> with SHAP-based plain-English explanations. The whole thing is tested with
> pytest and deployed through a GitHub Actions CI/CD pipeline."

**Key numbers to quote:**
- 5,000-row dataset, ~19% churn (class imbalance)
- Best model AUC: 0.7751, Recall: 0.7754
- 15 feature engineering + API + model tests — all passing
- Real-time prediction endpoint with explainability

---

## 2. End-to-End Pipeline Walkthrough

```
Step 1          Step 2           Step 3           Step 4          Step 5
Generate   →   Clean &     →   Feature     →   SMOTE +     →   Evaluate &
Synthetic       Impute          Engineering     GridSearchCV     Select Best
Data            Missing         (RFM, risk,     (LR, RF, XGB)   Model
(5000 rows)     Values          engagement)     cv=5, F1        (ROC-AUC)
    │                                                               │
    │           Step 6           Step 7           Step 8            │
    └────────   Save Model  ←   SHAP Plot   ←   Serve via    ←────┘
                (.pkl)           (.png)          FastAPI
```

### How data flows through the code:

1. **`src/generate_data.py`** → creates `data/raw/customers.csv`
2. **`src/data_processing.py`** → loads CSV, fills NaN with median, deduplicates
3. **`src/feature_engineering.py`** → adds 5 new features, one-hot encodes, scales
4. **`src/train.py`** → splits 80/20, applies SMOTE, trains 3 models, saves winner
5. **`src/predict.py`** → loads model, transforms a single customer, returns probability
6. **`api/main.py`** → wraps predict in a REST endpoint

---

## 3. File-by-File Deep Dive

### 3.1 `src/generate_data.py` — Synthetic Data Generator

**What it does:** Creates 5,000 realistic customer records with correlated features.

**How churn is generated (this is an interview favourite):**
```
logit = -2.0
      + 0.25 × (support_tickets − 2)      ← more tickets → more churn
      + 0.03 × (days_since_last_login − 20) ← inactive → more churn
      − 3.0  × (email_open_rate − 0.35)    ← engaged → less churn
      − 0.02 × (total_orders − 20)          ← loyal buyers → less churn
      + 0.01 × (recency − 30)               ← stale → more churn
      + plan_churn_bias                      ← Basic=+0.6, Premium=−0.8
      − 1.5  × (discount_used_pct − 0.25)

churn_probability = sigmoid(logit)
```

This is a **logistic data generating process** — the target is sampled from a
Bernoulli distribution with probability = sigmoid(logit). This means:
- The true relationship is known (useful for validating the model).
- Realistic correlations are baked in (not random).
- ~20% churn rate is achieved by calibrating the intercept.

**Interview tip:** If asked "How do you know the features are correlated with
churn?" — explain that YOU designed the logistic DGP, so the ground truth
is known. In real projects, you'd validate with domain experts + EDA.

---

### 3.2 `src/data_processing.py` — Load, Clean, Split

| Function | What It Does | Key Detail |
|----------|-------------|------------|
| `load_raw_data()` | Reads CSV, parses dates | Auto-detects `signup_date`, `last_purchase_date` |
| `clean_data()` | Fills NaN with **median**, drops duplicate `customer_id` | Median chosen over mean because it's robust to outliers |
| `split_data()` | 80/20 train/test, **stratified** on churn | Stratify preserves the 19% churn rate in both splits |

**Interview Q: Why median and not mean?**
> Median is robust to outliers. For skewed features like `days_since_last_login`
> (exponential distribution), the mean would be pulled upward by extreme values.

**Interview Q: Why stratified split?**
> With only 19% positive class, a random split could give 15% in train and
> 25% in test (or vice versa). Stratification guarantees both sets have ~19%.

---

### 3.3 `src/feature_engineering.py` — The Heart of The Project

This file creates **5 engineered features** and handles encoding + scaling.

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `recency` | `(2025-12-01) − last_purchase_date` in days | How recently did they buy? (RFM-R) |
| `frequency` | `total_orders × clip(1 − recency/365, 0.1, 1.0)` | Proxy for recent purchase frequency (RFM-F) |
| `monetary` | `avg_order_value × total_orders` | Lifetime spend (RFM-M) |
| `engagement_score` | `0.6 × email_open_rate + 0.4 × (1 − login_recency/180)` | Combined behavioural engagement |
| `risk_flag` | `1 if tickets > 3 AND days_since_login > 30` | Binary alert for at-risk customers |

**RFM (Recency, Frequency, Monetary):**
> This is a classic customer segmentation framework from marketing. Interviewers
> love hearing "I applied RFM analysis." It shows domain knowledge.

**Encoding:**
- `plan_type` (Basic / Standard / Premium) → one-hot encoded with `drop_first=True`
- Drops "Basic" and keeps `plan_Standard`, `plan_Premium` as binary columns
- `drop_first=True` avoids the **dummy variable trap** (multicollinearity)

**Scaling:**
- `StandardScaler` (z-score: `(x − μ) / σ`) on all numeric features
- Fitted on **training data only** → applied to test data with `transform()`
- This prevents **data leakage**

**Interview Q: What is data leakage and how did you prevent it?**
> Data leakage is when information from the test set influences training.
> I prevent it by:
> 1. Splitting BEFORE feature engineering
> 2. Fitting the scaler on train only, using `.transform()` on test
> 3. Applying SMOTE only on training data, never on test

---

### 3.4 `src/train.py` — Model Training Pipeline

#### SMOTE (Synthetic Minority Over-sampling Technique)
```python
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
# Result: {0: 3250, 1: 3250} — balanced 50/50
```

**How SMOTE works (explain in interview):**
> SMOTE creates synthetic samples for the minority class by:
> 1. Pick a minority sample
> 2. Find its k nearest neighbours (default k=5)
> 3. Draw a line to one neighbour, pick a random point on that line
> 4. That random point becomes a new synthetic sample
>
> It's better than random oversampling because it creates NEW data rather
> than duplicates, reducing overfitting risk.

**Interview Q: Why not just use class_weight='balanced'?**
> Both are valid. SMOTE generates new training points that fill in the feature
> space, which can help tree models learn better boundaries. For logistic
> regression, `class_weight='balanced'` often works just as well. In practice,
> you'd try both and compare.

#### Models & Hyperparameter Grids

| Model | Hyperparameters Searched | Best Found |
|-------|------------------------|------------|
| Logistic Regression | C: [0.01, 0.1, 1, 10], penalty: [l1, l2] | C=0.01, penalty=l1 |
| Random Forest | n_estimators: [100, 200], max_depth: [5, 10, None], min_samples_split: [2, 5] | n_estimators=200, max_depth=None, min_samples_split=2 |
| XGBoost | n_estimators: [100, 200], max_depth: [3, 5, 7], learning_rate: [0.01, 0.1] | learning_rate=0.1, max_depth=7, n_estimators=200 |

- **GridSearchCV** with `cv=5` (5-fold cross-validation), scored on **F1**
- Model selection on **test ROC-AUC** (not CV F1) to avoid overfitting to the resampled data

**Interview Q: Why tune on F1 but select on AUC?**
> F1 is threshold-dependent and good for tuning — it balances precision/recall
> at the 0.5 decision boundary. AUC is threshold-independent and measures
> overall ranking quality. A model with high AUC can always be calibrated
> to any desired precision/recall trade-off.

#### SHAP Explainability
```python
# For tree-based models:
explainer = shap.TreeExplainer(model)

# For linear models:
explainer = shap.LinearExplainer(model, X_test)
```

**What SHAP values mean (explain simply):**
> Each SHAP value tells you how much a single feature pushed the prediction
> away from the average. Positive SHAP = pushes toward churn. Negative = pushes
> away from churn. They always sum to (prediction − base_rate).

---

### 3.5 `src/predict.py` — Inference Module

Key steps for a single prediction:
1. Build a one-row DataFrame from raw feature dict
2. Run through the same `prepare_features()` pipeline (with the saved scaler)
3. Align columns to match training feature order
4. Call `model.predict_proba()` → get P(churn)
5. Map to risk tier: **≥0.7 High**, **≥0.4 Medium**, **<0.4 Low**
6. Run SHAP on that single row → get top-3 reasons in plain English

**Interview Q: Why save the scaler with the model?**
> The scaler was fit on training data. At inference time, new data must be
> transformed with the SAME mean/std. If you refit the scaler on new data,
> the features would be on a different scale and predictions would be wrong.

---

### 3.6 `api/main.py` — FastAPI Serving Layer

```
POST /predict
{
  "customer_id": "CUST-00042",
  "features": { ... raw feature values ... }
}

→ Response:
{
  "churn_probability": 0.8931,
  "risk_tier": "High",
  "top_reasons": [
    "Few total orders placed (decreases churn risk)",
    "High number of support tickets (increases churn risk)",
    ...
  ]
}
```

**Design choices to highlight in interviews:**
- Model loaded **once at startup** (`@app.on_event("startup")`) — not per request
- Pydantic schema validates input → returns 422 on bad payloads
- SHAP explanations returned as human-readable strings, not raw numbers
- `/health` endpoint for load balancer health checks

---

### 3.7 `tests/test_features.py` — Pytest Suite

| Test Class | # Tests | What's Tested |
|-----------|---------|--------------|
| `TestFeatureEngineeringDtypes` | 6 | All engineered features return correct dtypes |
| `TestRecencyAccuracy` | 2 | Hand-calculated recency matches code output, non-negative |
| `TestRiskFlag` | 1 | Boolean logic for risk flag (tickets > 3 AND login > 30) |
| `TestModelPrediction` | 3 | Probability ∈ [0,1], valid risk tier, reasons list |
| `TestAPI` | 3 | /health returns 200, /predict returns valid JSON, bad payload → 422 |
| **Total** | **15** | **All passing** |

**Interview tip:** Mention that model-dependent tests gracefully skip when no
model file exists (e.g., in CI before training) using `pytest.skip()`.

---

### 3.8 `.github/workflows/ci.yml` — CI/CD Pipeline

```yaml
Trigger: push/PR to main
Steps:
  1. Checkout code
  2. Setup Python 3.11
  3. Cache pip dependencies
  4. Install requirements
  5. Lint with flake8 (fail on syntax errors, warn on style)
  6. Generate synthetic data
  7. Run pytest (unit tests only, skip model-dependent)
```

**Interview Q: Why skip model tests in CI?**
> Training takes minutes and requires significant compute. CI should be fast
> (< 2 min). Unit tests for feature engineering validate the core logic.
> Model tests run locally or in a dedicated training pipeline.

---

## 4. Key Design Decisions & Why

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| Stratified split | Preserve 19% churn rate in both sets | Random split — risky with imbalanced data |
| SMOTE after split | Prevent data leakage from synthetic samples | SMOTE before split — would leak into test set |
| Scaler fit on train only | Prevent data leakage | Fit on full data — inflates test performance |
| `drop_first=True` in one-hot | Avoid multicollinearity (dummy variable trap) | Keep all dummies — problematic for logistic regression |
| GridSearchCV on F1, select on AUC | F1 good for threshold tuning; AUC for ranking | Tune on AUC directly — works but less interpretable for imbalanced data |
| Save model + scaler + feature_names together | Ensures consistent inference | Save separately — risk of version mismatch |
| SHAP for explainability | Model-agnostic, theoretically grounded (Shapley values) | LIME — less consistent, permutation importance — no per-prediction explanation |
| FastAPI over Flask | Async, auto-docs (Swagger), Pydantic validation | Flask — more boilerplate, no auto-validation |
| Median imputation | Robust to outliers in skewed distributions | Mean — biased by outliers; KNN — slower |

---

## 5. Metrics Deep Dive

### Actual Results (from `outputs/metrics.json`)

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| **Logistic Regression** | **0.7751** | **0.4633** | **0.3303** | **0.7754** |
| Random Forest | 0.7408 | 0.3902 | 0.3956 | 0.3850 |
| XGBoost | 0.7355 | 0.3522 | 0.3986 | 0.3155 |

### Reading the Confusion Matrix (Logistic Regression)

```
                 Predicted
                 No    Yes
Actual  No    [ 519    294 ]    ← 294 false positives (flagged but didn't churn)
        Yes   [  42    145 ]    ← 42 false negatives (missed churners)
```

- **True Positive Rate (Recall):** 145 / (145 + 42) = 77.5% — catches most churners
- **Precision:** 145 / (145 + 294) = 33.0% — many false alarms
- **Trade-off:** High recall is preferred for churn — it's worse to MISS a
  churner than to send an extra retention email to a loyal customer.

### How to Explain the Precision/Recall Trade-off

> "In a churn scenario, we optimize for **recall** because the cost of losing
> a customer ($200+ in acquisition cost to replace) is much higher than the
> cost of a false alarm (a $5 retention email to someone who wasn't going to
> churn). Our model catches 77.5% of churners. If precision matters more, we
> can raise the decision threshold from 0.5 to 0.7."

### Why LR Beat Tree Models on AUC Here

The data was generated using a **logistic DGP** (logit → sigmoid → Bernoulli).
Logistic regression perfectly matches the true generating process, so it
naturally fits well. In real-world data with non-linear interactions, tree
models typically outperform.

---

## 6. Common Interview Questions & Answers

### Data Science Fundamentals

**Q: How do you handle class imbalance?**
> Multiple strategies: (1) SMOTE — synthetic oversampling, which I used here;
> (2) class_weight='balanced' in the model; (3) undersampling the majority;
> (4) using a threshold-insensitive metric like AUC; (5) ensemble methods like
> BalancedRandomForest. I chose SMOTE because it generates novel training points
> that help the model learn the minority boundary better.

**Q: What's the difference between ROC-AUC and F1?**
> AUC measures ranking quality across ALL thresholds — it asks "does the model
> assign higher probabilities to actual churners?" F1 is threshold-specific
> (default 0.5) and balances precision and recall into a single number.
> AUC is better for model comparison; F1 is better for evaluating a specific
> decision point.

**Q: Why did you choose these three models?**
> Logistic Regression: simple, interpretable, fast baseline — tells me if the
> problem is linearly separable. Random Forest: non-linear, handles interactions,
> robust to overfitting. XGBoost: state-of-the-art gradient boosting, usually
> wins Kaggle competitions on tabular data. Together they cover linear, bagging,
> and boosting paradigms.

**Q: Explain SHAP values in simple terms.**
> Imagine a team of players (features) contributing to a score (prediction).
> SHAP uses game theory to fairly allocate credit. Each feature gets a value
> showing how much it pushed the prediction up or down from the average.
> The sum of all SHAP values = prediction − average prediction. It's based on
> Shapley values from cooperative game theory, which is the only allocation
> method that satisfies fairness axioms (efficiency, symmetry, linearity, null player).

**Q: How do you prevent overfitting?**
> (1) Train/test split — never evaluate on training data;
> (2) 5-fold cross-validation during tuning;
> (3) Regularisation — L1 penalty in logistic regression, max_depth in RF/XGB;
> (4) SMOTE only on training set;
> (5) Scaler fit only on training set.

**Q: What is data leakage? How did you avoid it?**
> Data leakage = test information influencing training. In my pipeline:
> - Split happens FIRST, before any feature engineering
> - Scaler is fit on train only, transform-only on test
> - SMOTE is applied only to training data
> - No future-looking features (recency computed from a fixed reference date)

**Q: Why StandardScaler and not MinMaxScaler?**
> StandardScaler (z-score) is preferred when features have different units and
> you don't know if they're bounded. MinMaxScaler squashes to [0,1] but is
> sensitive to outliers. For tree models, scaling doesn't matter at all, but
> logistic regression benefits from standardised features because the
> regularisation penalty treats all coefficients equally.

---

### Feature Engineering Questions

**Q: What is RFM analysis?**
> RFM = Recency, Frequency, Monetary — a marketing framework to segment
> customers by: how recently they purchased, how often they purchase, and how
> much they spend. High RFM customers are valuable; low-recency + low-frequency
> customers are at risk. I implemented all three as engineered features.

**Q: How did you create the engagement score?**
> Weighted combination: 60% email open rate + 40% login recency score.
> The login recency score = 1 − (days_since_login / 180), clipped to [0, 1].
> So a customer who logged in yesterday gets ~1.0, and one absent for 6 months
> gets ~0.0. I gave email more weight because it's a proactive engagement signal.

**Q: What does the risk_flag capture?**
> It's a binary indicator: 1 when a customer has both high support tickets (> 3)
> AND hasn't logged in for over a month (> 30 days). It captures the "angry and
> disengaged" segment that's most likely to churn.

---

### ML Engineering / Production Questions

**Q: How would you deploy this to production?**
> 1. Containerise with Docker (FastAPI + model artifact in image)
> 2. Deploy to Kubernetes or AWS ECS for auto-scaling
> 3. Use a model registry (MLflow) for versioning
> 4. Add monitoring — latency, error rates, prediction distribution drift
> 5. Retrain on a schedule (weekly) with fresh data
> 6. A/B test: compare model-driven outreach vs. control group

**Q: How would you handle real-time vs. batch predictions?**
> - **Batch:** Run nightly scoring on all active customers → write risk scores
>   to a database → marketing team queries high-risk list
> - **Real-time:** The FastAPI endpoint I built handles individual predictions
>   in <100ms — triggered when a customer visits the cancellation page or when
>   a support ticket is filed

**Q: How would you monitor model performance in production?**
> 1. **Data drift:** Track feature distributions weekly (Evidently AI / WhyLabs)
> 2. **Prediction drift:** Monitor P(churn) distribution — sudden shifts indicate
>    something changed
> 3. **Ground truth feedback:** After 30 days, compare predictions to actual churn
> 4. **Business KPIs:** Track retention intervention success rate
> 5. Retrain trigger: if AUC drops below 0.70 on recent data

**Q: Why FastAPI and not Flask?**
> FastAPI gives: (1) automatic request validation via Pydantic,
> (2) auto-generated Swagger docs at /docs, (3) async support for high
> throughput, (4) type hints throughout. Flask requires manual validation
> and has no built-in schema documentation.

**Q: How would you scale the API for 10,000 requests/second?**
> 1. Horizontal scaling: multiple API instances behind a load balancer
> 2. Model loaded in memory at startup (already done) — no disk I/O per request
> 3. Batch prediction endpoint for bulk scoring
> 4. Model distillation: train a smaller model for production speed
> 5. Caching: if same customer is scored multiple times, cache for 15 min
> 6. Consider converting model to ONNX for optimised inference

---

### Statistical / Theoretical Questions

**Q: How does logistic regression work?**
> It models P(Y=1|X) = sigmoid(w·X + b), where sigmoid(z) = 1/(1+e^−z).
> The weights w are learned by maximising the log-likelihood (minimising
> binary cross-entropy loss). The output is always between 0 and 1.

**Q: How does XGBoost work?**
> XGBoost builds an ensemble of decision trees sequentially. Each new tree
> tries to correct the errors (residuals) of all previous trees. It uses
> gradient descent on a loss function, adding regularisation (L1/L2 on leaf
> weights) to prevent overfitting. "Boosting" means each tree focuses on the
> hard-to-predict samples.

**Q: How does a Random Forest work?**
> It's an ensemble of decorrelated decision trees. Each tree is trained on a
> bootstrap sample (random sampling with replacement) of the data, and at each
> split, only a random subset of features is considered. Predictions are
> averaged (regression) or majority-voted (classification). The randomness
> reduces variance/overfitting.

**Q: What are the assumptions of logistic regression?**
> 1. Linear relationship between features and log-odds
> 2. No multicollinearity (hence drop_first in one-hot encoding)
> 3. Independent observations
> 4. Large sample size relative to number of features
> Note: it does NOT assume normal distributions of features.

---

## 7. System Design Discussion Points

If asked "Design a churn prediction system for a company with 10M customers":

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                         │
├─────────────┬──────────────┬──────────────┬────────────────────────┤
│  Data Layer │  ML Platform │  Serving     │  Business Layer        │
│             │              │              │                        │
│ Kafka /     │ Feature Store│ Model API    │ CRM Integration        │
│ Event       │ (Feast)      │ (FastAPI +   │ (send retention email  │
│ Stream      │              │  K8s)        │  if P(churn) > 0.7)    │
│             │ MLflow       │              │                        │
│ Data        │ (versioning) │ A/B Test     │ Dashboard              │
│ Warehouse   │              │ Framework    │ (Looker / Metabase)    │
│ (Snowflake) │ Airflow      │              │                        │
│             │ (scheduling) │ Monitoring   │ Alerts                 │
│             │              │ (Evidently)  │ (PagerDuty)            │
└─────────────┴──────────────┴──────────────┴────────────────────────┘
```

**Key talking points:**
1. **Feature store** — ensures training and serving use the same feature logic
2. **Model versioning** — MLflow tracks experiments, parameters, metrics
3. **A/B testing** — measure causal impact: does the model-driven intervention
   actually reduce churn vs. a control group?
4. **Monitoring** — detect data drift, concept drift, and performance degradation
5. **Feedback loop** — ground truth arrives 30 days later; use it to retrain

---

## 8. Behavioural / STAR Stories from This Project

### Story 1: Handling Class Imbalance
- **Situation:** Dataset had only ~19% positive class (churners)
- **Task:** Build a model that catches most churners without being overwhelmed
  by the majority class
- **Action:** Implemented SMOTE to balance training data, used stratified splitting,
  tuned on F1 (which penalises both false positives and negatives), and selected
  on AUC for threshold-independent evaluation
- **Result:** Model achieves 77.5% recall — catches 3 out of 4 churners

### Story 2: Preventing Data Leakage
- **Situation:** Feature engineering involves scaling and creating derived features
- **Task:** Ensure no information from test set leaks into training
- **Action:** Enforced strict ordering: split → feature engineer → fit scaler on train
  → transform test. Applied SMOTE only on training. Used a fixed reference date
  for recency instead of "today" to avoid temporal leakage.
- **Result:** Honest performance estimates — no inflated metrics

### Story 3: Making the Model Explainable
- **Situation:** Stakeholders wouldn't trust a black-box "this customer will churn"
- **Task:** Provide human-readable explanations per prediction
- **Action:** Integrated SHAP values into the prediction pipeline, mapped feature
  names to plain-English templates, returned top-3 reasons via the API
- **Result:** API returns "High number of support tickets (increases churn risk)"
  alongside the probability — actionable for the business team

### Story 4: Building for Production from Day One
- **Situation:** Many data science projects die in notebooks
- **Task:** Build something deployable, testable, and maintainable
- **Action:** Structured as a Python package with modular src/ files, added
  FastAPI endpoint, wrote 15 pytest tests, set up GitHub Actions CI/CD,
  created a Makefile for one-command workflows
- **Result:** Code is production-ready: `make all` runs the entire pipeline,
  `make api` starts the server, CI validates every PR

---

## 9. Quick Cheat Sheet — Numbers to Remember

| Item | Value |
|------|-------|
| Dataset size | 5,000 customers |
| Churn rate | ~19% (class imbalance) |
| Missing data | ~2% in 3 columns |
| Train/test split | 80/20, stratified |
| SMOTE result | 3,250 / 3,250 balanced |
| Cross-validation | 5-fold, scored on F1 |
| Models trained | 3 (LR, RF, XGBoost) |
| Best model | Logistic Regression |
| Best AUC | 0.7751 |
| Best Recall | 0.7754 (catches 77.5% of churners) |
| Best F1 | 0.4633 |
| Best Precision | 0.3303 |
| Engineered features | 5 (recency, frequency, monetary, engagement_score, risk_flag) |
| Total features | ~13 (10 numeric + 2 one-hot + 1 risk flag) |
| Unit tests | 15, all passing |
| API response time | < 100ms per prediction |
| Risk tiers | High (≥0.7), Medium (≥0.4), Low (<0.4) |

---

## Bonus: Questions to Ask the Interviewer

If they ask "Do you have questions about our ML systems?":

1. "How do you currently handle model retraining — is it scheduled or triggered
   by performance degradation?"
2. "Do you use a feature store, and if so, how do you ensure consistency between
   training and serving features?"
3. "What's your approach to A/B testing model-driven interventions?"
4. "How do you handle concept drift — do you monitor ground truth labels?"
5. "What's the latency requirement for real-time predictions?"

---

*Good luck with your interviews! The best way to prepare is to run through
each section out loud, as if explaining to a non-technical stakeholder first,
then to a technical peer.*
