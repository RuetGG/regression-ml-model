import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_models(pipelines, X_train, X_test, y_train, y_test):
    results = []

    # Invert log transform on test targets to get actual dollar values
    y_train_dollars = np.expm1(y_train)
    y_test_dollars = np.expm1(y_test)

    for name, pipeline in pipelines.items():
        # Get log predictions
        train_preds_log = pipeline.predict(X_train)
        test_preds_log = pipeline.predict(X_test)

        # Convert predictions to real dollar values
        train_preds_dollars = np.expm1(train_preds_log)
        test_preds_dollars = np.expm1(test_preds_log)

        # Train Metrics (in dollars & log R2)
        train_r2 = r2_score(y_train, train_preds_log)

        # Test Metrics (MAE, MSE, RMSE in actual Dollars)
        test_mae = mean_absolute_error(y_test_dollars, test_preds_dollars)
        test_mse = mean_squared_error(y_test_dollars, test_preds_dollars)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, test_preds_log)

        results.append(
            {
                "Model": name,
                "Test MAE ($)": round(test_mae, 2),
                "Test MSE ($)": round(test_mse, 2),
                "Test RMSE ($)": round(test_rmse, 2),
                "Test R2": round(test_r2, 4),
                "Train R2": round(train_r2, 4),  # Check for overfitting
            }
        )

    # Convert to DataFrame and sort by best R2 score
    metrics_df = pd.DataFrame(results).sort_values(
        by="Test R2", ascending=False
    )
    return metrics_df


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from src.train import train_models

    # Train models and run evaluation
    pipelines, summary, (X_train, X_test, y_train, y_test) = train_models(
        "data/used_cars_clean_fe.csv"
    )

    metrics_df = evaluate_models(
        pipelines, X_train, X_test, y_train, y_test
    )

    print("\n--- Model Evaluation Results ---")
    print(metrics_df.to_string(index=False))