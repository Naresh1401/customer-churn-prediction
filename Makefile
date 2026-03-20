.PHONY: install data train test lint api clean all

# ── Setup ──────────────────────────────────────────────────
install:
	pip install --upgrade pip && pip install -r requirements.txt

# ── Data ───────────────────────────────────────────────────
data:
	python -m src.generate_data

# ── Training ───────────────────────────────────────────────
train: data
	python -m src.train

# ── Testing & Linting ─────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ api/ tests/ --max-line-length=120 --statistics

# ── API ────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Cleanup ────────────────────────────────────────────────
clean:
	rm -rf data/raw/* data/processed/* models/* outputs/* __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Full Pipeline ─────────────────────────────────────────
all: install data train test
