import streamlit as st, requests

st.title("Ames House Price – Mini")
api = st.secrets.get("API_URL", "http://localhost:8000")

def num(name, val):
    return st.number_input(name, value=val)

vals = {
    "OverallQual": int(num("OverallQual (1–10)", 6)),
    "GrLivArea": num("GrLivArea (sqft)", 1500.0),
    "GarageCars": num("GarageCars", 2.0),
    "TotalBsmtSF": num("TotalBsmtSF", 800.0),
    "FullBath": num("FullBath", 2.0),
    "YearBuilt": int(num("YearBuilt", 1995)),
    "1stFlrSF": num("1stFlrSF", 900.0),
    "LotArea": num("LotArea", 8000.0),
    "Neighborhood": st.text_input("Neighborhood", "CollgCr"),
    "HouseStyle": st.text_input("HouseStyle", "1Story"),
    "KitchenQual": st.text_input("KitchenQual (Ex/Gd/TA/Fa)", "Gd"),
    "MSZoning": st.text_input("MSZoning (RL/RM/…)", "RL")
}

if st.button("Predict"):
    r = requests.post(f"{api}/predict", json=vals, timeout=10)
    st.write(r.json())
