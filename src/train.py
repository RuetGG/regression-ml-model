import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def get_preprocessor(scale_numeric=True):
    categorical_features=[
        'brand',
        'fuel_type',               
        'transmission',           
        'accident',                
        'clean_title',
        'num_gears',
        'transmission_type'             
    ]
    
    numeric_features = [
        "milage",
        "model_year",
        "car_age",
        "engine_hp",
        "engine_size_l",
        "engine_cylinders",
    ]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_transformer = Pipeline(steps=num_steps)

    cat_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="unknown"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )


def train_models(data_path="data/used_cars_clean_fe.csv"):
    df = pd.read_csv(data_path)

    # Separate target and features
    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define all 7 regressors and whether they require scaling
    models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Lasso Regression": (Lasso(alpha=0.1), True),
        "Support Vector Regressor": (SVR(C=1.0), True),
        "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42), False),
        "Decision Tree": (DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42), False),
    }

    trained_pipelines = {}
    results = []

    for name, (model, scale_numeric) in models.items():
        preprocessor = get_preprocessor(scale_numeric=scale_numeric)
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )

        # Fit and measure training time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        trained_pipelines[name] = pipeline
        results.append({"Model": name, "Train Time (s)": round(elapsed_time, 4)})
        print(f"[{name}] Trained in {elapsed_time:.4f}s")

    summary_df = pd.DataFrame(results)
    return trained_pipelines, summary_df, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    pipelines, summary, _ = train_models()
    print("\n--- Training Summary ---")
    print(summary)