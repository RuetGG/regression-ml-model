import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance


def get_preprocessor(scale_numeric=True):
    categorical_features=[
        'brand',
        'fuel_type',               
        'accident',                
        'clean_title',
        'transmission_type'             
    ]
    
    numeric_features = [
        "milage",
        "model_year",
        'milage_per_year',
        'num_gears',
        "car_age",
        "engine_hp",
        "engine_size_l",
        "engine_cylinders",
        "luxury_brand", 
        'is_turbo'
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


def train_models(data_path="../data/used_cars_clean_fe.csv",
                 model_output="../models/best_model.pkl",
                 metrics_output="../outputs/metrics/model_comparison.csv"):
    
    print("=" * 65)
    print(" AUTOMATED MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 65)
    
    df = pd.read_csv(data_path)

    # Feature & Target Selection
    X = df.drop(columns=["price"])
    y_raw = df["price"]
    y_log = np.log1p(y_raw)

    # Train/Test Split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_real = np.expm1(y_test_log)

    # Regressors dictionary: (Estimator, Scale Numeric Features)
    models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=10.0), True),
        "Lasso Regression": (Lasso(alpha=0.01, max_iter=10000), True),
        "Support Vector Regressor": (SVR(C=1.0, epsilon=0.1), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42), False)
    }

    results = []
    best_score = -float("inf")
    best_pipeline = None

    for name, (model, scale_numeric) in models.items():
        preprocessor = get_preprocessor(scale_numeric=scale_numeric)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train_log)
        train_time = time.time() - start_time

        # Predict (Log Scale)
        t0 = time.time()
        preds_log = pipeline.predict(X_test)
        pred_time = time.time() - t0

        # Predict (Real Dollar Scale)
        preds_real = np.expm1(preds_log)

        # Metrics Computation
        r2_log = r2_score(y_test_log, preds_log)
        mae_real = mean_absolute_error(y_test_real, preds_real)
        mse_real = mean_squared_error(y_test_real, preds_real)
        rmse_real = np.sqrt(mse_real)

        results.append({
            "Model": name,
            "Test R2": round(r2_log, 4),
            "Test MAE ($)": round(mae_real, 2),
            "Test MSE ($)": round(mse_real, 2),
            "Test RMSE ($)": round(rmse_real, 2),
            "Train Time (s)": round(train_time, 4),
            "Pred Time (s)": round(pred_time, 4)
        })

        print(f"[{name}] Completed in {train_time:.2f}s | Test R2: {r2_log:.4f} | MAE: ${mae_real:,.2f}")

        # Track best model based on log-scale R2 stability
        if r2_log > best_score:
            best_score = r2_log
            best_pipeline = pipeline

    # Export Metrics Table
    results_df = pd.DataFrame(results).sort_values(by="Test R2", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("FINAL MODEL EVALUATION SUMMARY")
    print("=" * 65)
    print(results_df.to_string(index=False))

    # Save Pipeline and Metrics Artifacts
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)


    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    importance = permutation_importance(
        best_pipeline,
        X_test,
        y_test_log,
        n_repeats=5,
        random_state=42,
        scoring="r2"
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance.importances_mean
    }).sort_values("Importance", ascending=False)

    print("\nFEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))
    
    joblib.dump(best_pipeline, model_output)
    results_df.to_csv(metrics_output, index=False)

    print(f"\n[ARTIFACT] Saved Best Pipeline -> {model_output}")
    print(f"[ARTIFACT] Saved Benchmark CSV  -> {metrics_output}")

if __name__ == "__main__":
    train_models()