# DEPLOYMENT.md — How to Run the MVP

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/bvoncarr/world-happiness-predictor
cd world-happiness-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset (requires Kaggle account)
```bash
kaggle datasets download -d elvisbui/world-happiness-report-2005-2025-panel
unzip world-happiness-report-2005-2025-panel.zip -d data/raw/
```

### 4. Run data pipeline
```bash
python src/data_processing.py
```

### 5. Run notebooks in order
```bash
python -m jupyter notebook
```
Open and run:
- `notebooks/01_eda.ipynb`
- `notebooks/02_baseline.ipynb`
- `notebooks/03_experiments.ipynb`

### 6. Index RAG knowledge base
```bash
python src/rag.py
```

### 7. Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Demo Mode

The app runs in demo mode if no trained model is found.
It uses a simple weighted sum to estimate happiness scores.
Good for quick UI demos without running the full pipeline.

---

## File Size Notes

- `models/baseline.pkl` — ~1MB
- `models/improved.pkl` — ~50MB
- `data/raw/` — ~47KB
- `.chromadb/` — ~5MB

Add to `.gitignore`:
data/raw/
.chromadb/
pycache/
*.pyc
.env
.DS_Store

---

## Cloud Deployment (Hugging Face Spaces)

1. Create a new Space at huggingface.co/spaces
2. Select Streamlit as the SDK
3. Push your repo
4. Add dataset download step to `setup.sh`
5. Set any API keys in Space secrets
