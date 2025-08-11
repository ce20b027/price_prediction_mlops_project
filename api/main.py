from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Ames House Price API")

pre = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/model.pkl")
meta = joblib.load("models/feature_meta.pkl")

class AmesFeatures(BaseModel):
    OverallQual: int = Field(..., ge=1, le=10)
    GrLivArea: float
    GarageCars: float
    TotalBsmtSF: float
    FullBath: float
    YearBuilt: int
    FirstFlrSF: float = Field(alias="1stFlrSF")
    LotArea: float
    Neighborhood: str
    HouseStyle: str
    KitchenQual: str
    MSZoning: str

@app.post("/predict")
def predict(f: AmesFeatures):
    row = {
        "OverallQual": f.OverallQual,
        "GrLivArea": f.GrLivArea,
        "GarageCars": f.GarageCars,
        "TotalBsmtSF": f.TotalBsmtSF,
        "FullBath": f.FullBath,
        "YearBuilt": f.YearBuilt,
        "1stFlrSF": f.FirstFlrSF,
        "LotArea": f.LotArea,
        "Neighborhood": f.Neighborhood,
        "HouseStyle": f.HouseStyle,
        "KitchenQual": f.KitchenQual,
        "MSZoning": f.MSZoning
    }
    X = pd.DataFrame([row])
    Xp = pre.transform(X)
    price = float(model.predict(Xp)[0])
    return {"predicted_price": price}
