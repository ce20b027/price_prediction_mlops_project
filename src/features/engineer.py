import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# A compact, high-signal subset of features
NUM_COLS = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
    "FullBath", "YearBuilt", "1stFlrSF", "LotArea"
]
CAT_COLS = [
    "Neighborhood", "HouseStyle", "KitchenQual", "MSZoning"
]

def main():
    in_path = Path("data/processed/cleaned.csv")
    df = pd.read_csv(in_path)

    keep = NUM_COLS + CAT_COLS + ["SalePrice"]
    df = df[keep].copy()

    # Build preprocessors
    num_pipe = Pipeline(steps=[("scale", StandardScaler())])
    try:
        # scikit-learn >=1.2
        cat_pipe = Pipeline(steps=[("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    except TypeError:
        # fallback for older versions
        cat_pipe = Pipeline(steps=[("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop"
    )

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    Xp = preproc.fit_transform(X)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(Xp).to_csv("data/processed/featured.csv", index=False)
    y.to_csv("data/processed/target.csv", index=False)
    joblib.dump(preproc, "models/preprocessor.pkl")

    # Save meta
    meta = {
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
    }
    joblib.dump(meta, "models/feature_meta.pkl")

    print("Saved features, target, and preprocessor.")

if __name__ == "__main__":
    main()
