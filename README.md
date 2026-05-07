# 🌍 World Happiness Predictor

A machine learning web app that predicts a country's happiness score based on 6 key factors from the World Happiness Report (2005–2025), with RAG-powered research insights.

---

## Problem Statement

What makes countries happy? This project uses 20 years of global happiness data to predict happiness scores and identify the most influential factors — from GDP to social support to corruption levels.

---

## Dataset

- **Source:** [World Happiness Report 2005–2025 — Kaggle](https://www.kaggle.com/datasets/elvisbui/world-happiness-report-2005-2025-panel)
- **Size:** 2,116 rows · 168 countries · 2011–2025
- **Target:** `happiness_score` (Life Ladder score)
- **Features:** GDP per capita, social support, healthy life expectancy, freedom, generosity, corruption
- **Split:** 80% train / 20% test

---

## Methodology

### Pipeline
Raw Data → Cleaning → Feature Engineering → Model Training → RAG Insights → Streamlit App

### Models

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Baseline: Linear Regression | ~0.15 | ~0.20 | ~0.92 |
| Improved: Random Forest | ~0.08 | ~0.12 | ~0.97 |

### RAG Component
- 10 curated research summaries indexed in **ChromaDB**
- Embeddings via `all-MiniLM-L6-v2`
- Top-3 most relevant insights retrieved per prediction
- Sources: World Happiness Report, WHO, Transparency International

---

## Project Structure
world-happiness-predictor/
├── app.py                  # Streamlit MVP
├── requirements.txt
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Train/test splits
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_baseline.ipynb   # Linear Regression
│   └── 03_experiments.ipynb # Random Forest + comparison
├── src/
│   ├── data_processing.py  # Cleaning & splitting
│   ├── model.py            # Baseline & improved models
│   ├── rag.py              # ChromaDB RAG pipeline
│   └── utils.py            # Plotting utilities
└── models/                 # Saved model weights

---

## Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/bvoncarr/world-happiness-predictor
cd world-happiness-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
kaggle datasets download -d elvisbui/world-happiness-report-2005-2025-panel
unzip world-happiness-report-2005-2025-panel.zip -d data/raw/

# 4. Run data pipeline
python src/data_processing.py

# 5. Run notebooks in order
python -m jupyter notebook

# 6. Index RAG knowledge base
python src/rag.py

# 7. Launch the app
streamlit run app.py
```

---

## MVP Features

- 🎛️ Interactive sliders to simulate any country's conditions
- 🔮 Real-time happiness score prediction
- 🌐 Most similar real country matched to your prediction
- 📊 Factor breakdown chart showing each driver's contribution
- 📚 RAG-powered research insights from global happiness studies
- 🏆 Live top 5 happiest countries from latest data

---

## Key Findings

- **GDP per capita** is the strongest single predictor of happiness
- **Social support** is the second most important factor
- **Nordic countries** (Finland, Iceland, Denmark) consistently top the rankings
- **Random Forest** captures non-linear relationships, significantly outperforming Linear Regression
- Countries with high happiness share strong institutions, low corruption, and robust social safety nets

---

## Future Work

- Add time series forecasting (predict future happiness trends)
- Include more features (inequality, climate, urbanization)
- Deploy to Hugging Face Spaces
- Add country-to-country comparison feature

---

## Data Source

World Happiness Report 2005–2025 · Kaggle · Elvis Bui
