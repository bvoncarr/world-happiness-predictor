# EXPERIMENTS.md — Model Experiments & Results

## Dataset
- **Source:** World Happiness Report 2005–2025 (Kaggle / Elvis Bui)
- **Size:** 2,116 rows · 168 countries · 2011–2025
- **Target:** happiness_score (Life Ladder, scale 0–10)
- **Split:** 80% train (1,692 rows) / 20% test (424 rows)
- **Missing values:** Filled with column median

---

## Experiment 1: Linear Regression (Baseline)

| Hyperparameter | Value |
|----------------|-------|
| fit_intercept | True |
| normalize | False |

**Results (Test Set):**

| Metric | Value |
|--------|-------|
| MAE | ~0.15 |
| RMSE | ~0.20 |
| R² | ~0.92 |

**Observations:**
- Strong baseline — 92% of variance explained
- Assumes linear relationship between features and happiness
- GDP and social support have highest coefficients
- Struggles with non-linear interactions at extremes

---

## Experiment 2: Random Forest (Improved)

| Hyperparameter | Value |
|----------------|-------|
| n_estimators | 200 |
| max_depth | 10 |
| random_state | 42 |
| n_jobs | -1 |

**Results (Test Set):**

| Metric | Value |
|--------|-------|
| MAE | ~0.08 |
| RMSE | ~0.12 |
| R² | ~0.97 |

**Feature Importance (ranked):**
1. explained_log_gdp_per_capita
2. explained_social_support
3. explained_healthy_life_expectancy
4. explained_freedom
5. explained_corruption
6. explained_generosity

---

## Comparison

| Model | MAE | RMSE | R² | Training Time |
|-------|-----|------|----|---------------|
| Linear Regression (baseline) | ~0.15 | ~0.20 | ~0.92 | <1s |
| Random Forest (improved) | ~0.08 | ~0.12 | ~0.97 | ~5s |

**Winner: Random Forest** — +5% R², MAE cut nearly in half

---

## Error Analysis

**Residual Analysis:**
- Average prediction error: ~0.08 happiness points
- Largest errors occur at extreme scores (very happy >7.5 or very unhappy <3.5)
- Countries with unusual combinations (high GDP but low freedom) are hardest to predict
- Afghanistan and Lebanon show highest residuals due to rapid year-over-year changes

**Systematic patterns:**
- Model slightly underestimates very happy countries (ceiling effect)
- Model slightly overestimates very unhappy countries (floor effect)
- Both are expected behaviors of ensemble tree models

---

## RAG Evaluation

| Metric | Value |
|--------|-------|
| Knowledge base size | 10 research summaries |
| Embedding model | all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Retrieval speed | ~120ms |
| Sources | WHR, WHO, Transparency International |
