from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
import numpy as np


# =====================================
# FASTAPI APP
# =====================================

app = FastAPI()


# =====================================
# LOAD TRAINED MODEL
# =====================================

model = joblib.load(
    'sales_prediction_model_poly.pkl'
)


# =====================================
# REQUEST SCHEMA
# =====================================

class SalesInput(BaseModel):

    unit_price: float
    quantity: int
    brand: str
    category: str


# =====================================
# HOME ROUTE
# =====================================

@app.get("/")

def home():

    return {"message": "CI/CD Success"}

# =====================================
# PREDICTION ROUTE
# =====================================

@app.post("/predict")

def predict(data: SalesInput):

    # Convert input to dataframe
    df = pd.DataFrame([{
        'unit_price': data.unit_price,
        'quantity': data.quantity,
        'brand': data.brand,
        'category': data.category
    }])

    # Predict log value
    pred_log = model.predict(df)

    # Convert back to original scale
    pred = np.expm1(pred_log)

    # Return prediction
    return {
        'Predicted_Sales': round(float(pred[0]), 2)
    }