import joblib
import numpy as np
import pandas as pd

model = joblib.load('sales_prediction_model_poly.pkl')

new_data = pd.DataFrame({
    'unit_price': [300],
    'quantity': [5],
    'brand': ['Brand 1'],
    'category': ['Sports']
})

pred_log = model.predict(new_data)
pred = np.expm1(pred_log)

print(f"Predicted Sales: {pred[0]:.2f}")
