# Ames House Price – Mini (End-to-End MLOps project)

A tiny, reproducible end-to-end project that predicts **house sale price** using the **Ames Housing** dataset.  
It mirrors a typical MLOps flow:

1. **Data** → fetch & clean from OpenML
2. **Features** → numeric scale + categorical one-hot
3. **Training** → RandomForest with metrics tracked in **MLflow**
4. **Serving** → **FastAPI** endpoint for predictions
5.  **Streamlit** UI to try inputs without code


---

## Quickstart

### 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Start MLflow UI (Docker required)
```bash
cd deployment/mlflow
docker compose -f mlflow-docker-compose.yml up -d
# open http://localhost:5555
cd ../../
```

### 2) Data → Features → Train
```bash
python src/data/run_processing.py
python src/features/engineer.py
python src/models/train_model.py --config configs/model.yaml
```

**What you get:**
- `data/processed/cleaned.csv`: cleaned raw dataset
- `data/processed/featured.csv`: model-ready features
- `data/processed/target.csv`: target values
- `models/preprocessor.pkl` and `models/model.pkl`: saved artifacts
- MLflow run with params + metrics (R2, RMSE)

### 3) Serve the API
```bash
uvicorn api.main:app --reload --port 8000
```

**Sample cURL**
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "FullBath": 2,
    "YearBuilt": 2005,
    "1stFlrSF": 950,
    "LotArea": 8500,
    "Neighborhood": "CollgCr",
    "HouseStyle": "1Story",
    "KitchenQual": "Gd",
    "MSZoning": "RL"
  }'
```

### 4) (Optional) Streamlit UI
```bash
streamlit run streamlit_app/app.py
```

---

## Project Structure
```
.
├─ configs/
│  └─ model.yaml
├─ src/
│  ├─ data/run_processing.py
│  ├─ features/engineer.py
│  └─ models/train_model.py
├─ api/main.py
├─ streamlit_app/app.py
├─ deployment/mlflow/mlflow-docker-compose.yml
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## What is this project doing (plain English)

- It’s a **smart calculator** for house prices.
- We **train** it using thousands of past house sales in Ames (with their features).
- Once trained, you can **send house details** to the API and it returns a **price estimate**.
- We also **track experiments** using MLflow, so you always know which settings worked best.

---

## Notes
- First run downloads the dataset from OpenML (needs internet).
- You can tweak model params in `configs/model.yaml`.
- Extend features in `src/features/engineer.py` to improve accuracy.
