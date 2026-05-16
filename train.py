import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(
    "Sales Prediction"
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PolynomialFeatures
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv('retail_sales_dataset.csv')

# =====================================
# TARGET TRANSFORM
# =====================================

df['sales_amount_log'] = np.log1p(
    df['sales_amount']
)

# =====================================
# FEATURES & TARGET
# =====================================

X = df[
    ['unit_price', 'quantity', 'brand', 'category']
]

y = df['sales_amount_log']

# =====================================
# TRAIN TEST SPLIT
# =====================================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# PREPROCESSOR
# =====================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            'onehot',
            OneHotEncoder(handle_unknown='ignore'),
            ['brand', 'category']
        ),

        (
            'scaler',
            StandardScaler(),
            ['unit_price']
        )

    ],

    remainder='passthrough'
)

# =====================================
# PIPELINE
# =====================================

pipeline = Pipeline([

    (
        'preprocessor',
        preprocessor
    ),

    (
        'poly',
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),

    (
        'model',
        LinearRegression()
    )

])

# =====================================
# START MLFLOW
# =====================================

mlflow.start_run()

# =====================================
# LOG PARAMETERS
# =====================================

mlflow.log_param(
    'model_type',
    'Polynomial Regression'
)

mlflow.log_param(
    'degree',
    2
)

mlflow.log_param(
    'features',
    ['unit_price', 'quantity', 'brand', 'category']
)

# =====================================
# TRAIN MODEL
# =====================================

pipeline.fit(
    x_train,
    y_train
)

# =====================================
# PREDICTION
# =====================================

y_pred = pipeline.predict(
    x_test
)

# =====================================
# EVALUATION
# =====================================

r2 = r2_score(
    y_test,
    y_pred
)

rmse = root_mean_squared_error(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

print(f"R2 Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# =====================================
# LOG METRICS
# =====================================

mlflow.log_metric(
    'r2_score',
    r2
)

mlflow.log_metric(
    'rmse',
    rmse
)

mlflow.log_metric(
    'mae',
    mae
)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    pipeline,
    'sales_prediction_model_poly.pkl'
)

print("\nModel Saved Successfully")

# =====================================
# LOG MODEL TO MLFLOW
# =====================================

mlflow.sklearn.log_model(
    pipeline,
    artifact_path='model'
)

# =====================================
# END MLFLOW RUN
# =====================================

mlflow.end_run()

print("MLflow Tracking Completed")