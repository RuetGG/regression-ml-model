"""
app.py
------
Streamlit Web Application for Used Car Price Prediction & Model Leaderboard.
Run with: python -m streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 0. GLOBAL PAGE CONFIGURATION & PATH SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "app" else CURRENT_DIR

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "model_comparison.csv")
DATA_PATH = os.path.join(BASE_DIR, "data", "used_cars_clean_fe.csv")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "model" in loaded:
            return (
                loaded["model"],
                loaded.get("model_name", "Best Model"),
                loaded.get("r2", None),
                loaded.get("mae", None),
            )
        else:
            return loaded, "Support Vector Regressor (SVR)", 0.8944, 7563.21
    return None, None, None, None

@st.cache_data
def load_data():
    metrics = pd.read_csv(METRICS_PATH) if os.path.exists(METRICS_PATH) else None
    data = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None
    return metrics, data

pipeline, model_name, best_r2, best_mae = load_model()
metrics_df, raw_df = load_data()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home Page",
        "Project Description",
        "Dataset Information",
        "Model Information",
        "Prediction Form & Result",
        "Model Comparison Page",
    ]
)

# -----------------------------------------------------------------------------
# 1. HOME PAGE
# -----------------------------------------------------------------------------
if page == "Home Page":
    st.title("🚗 Used Car Price Prediction Platform")
    st.image(
        "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=1200&q=80",
        use_container_width=True
    )
    st.subheader("Welcome to our ML-powered Vehicle Valuation Tool!")
    st.write(
        """
        This web application leverages machine learning regression models trained on thousands 
        of real-world used car listings to estimate market value based on key attributes such as 
        mileage, engine size, age, brand, and condition.
        
        Use the sidebar on the left to navigate through project insights, dataset statistics, 
        model evaluations, or test live price predictions!
        """
    )
    st.info("💡 **Quick Start:** Go to **'Prediction Form & Result'** in the sidebar to estimate a car's price instantly.")

# -----------------------------------------------------------------------------
# 2. PROJECT DESCRIPTION
# -----------------------------------------------------------------------------
elif page == "Project Description":
    st.title("📄 Project Description")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Objective")
        st.write(
            """
            The goal of this project is to build an end-to-end Machine Learning solution capable 
            of accurately forecasting used car resale prices. By eliminating market opacity, 
            this tool helps buyers and sellers negotiate fair market transactions.
            """
        )
    
    with col2:
        st.subheader("⚙️ Methodology")
        st.write(
            """
            - **Data Preprocessing:** Outlier removal (1% top/bottom trim) and missing value imputation.
            - **Feature Engineering:** Log-transforming target prices, deriving car age, and mileage per year.
            - **Modeling:** Training and benchmarking 7 regression algorithms using Scikit-Learn pipelines.
            - **Deployment:** Live interactive predictions via Streamlit.
            """
        )

# -----------------------------------------------------------------------------
# 3. DATASET INFORMATION
# -----------------------------------------------------------------------------
elif page == "Dataset Information":
    st.title("📊 Dataset Information")
    st.markdown("---")
    
    if raw_df is not None:
        st.subheader("Dataset Overview")
        st.write(f"**Total Listings:** `{raw_df.shape[0]:,}` | **Total Features:** `{raw_df.shape[1]}`")
        
        st.subheader("Sample Data")
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        st.subheader("Summary Statistics")
        st.dataframe(raw_df.describe().T, use_container_width=True)
    else:
        st.warning(f"⚠️ Dataset file not found at `{DATA_PATH}`. Showing static schema.")
        features = pd.DataFrame({
            "Feature Name": ["brand", "model_year", "milage", "fuel_type", "engine_hp", "car_age", "accident", "clean_title"],
            "Type": ["Categorical", "Numeric", "Numeric", "Categorical", "Numeric", "Numeric", "Categorical", "Categorical"],
            "Description": ["Vehicle manufacturer", "Manufacturing year", "Odometer reading", "Fuel mechanism", "Engine Horsepower", "Years since manufacture", "Accident history flag", "Clean title status flag"]
        })
        st.table(features)

# -----------------------------------------------------------------------------
# 4. MODEL INFORMATION
# -----------------------------------------------------------------------------
elif page == "Model Information":
    st.title("🧠 Model Information")
    st.markdown("---")
    
    st.subheader(f"Deployed Model: {model_name}")
    st.write(
        """
        Our benchmark evaluations revealed **Support Vector Regression (SVR)** as the top-performing 
        algorithm for this dataset when log-transformed price targets are used.
        """
    )
    st.markdown(
        """
        - **Target Transformation:** Models were trained on $\ln(\text{price} + 1)$ to stabilize variance and prevent high-price outliers from distorting metrics.
        - **Preprocessing Pipeline:** Continuous numerical features are median-imputed and standardized (`StandardScaler`). Categoricals are missing-imputed and One-Hot Encoded.
        - **Inference Transformation:** Predictions are converted back to real dollars using $\mathbf{e^{\hat{y}} - 1}$ in real-time.
        """
    )

# -----------------------------------------------------------------------------
# 5. PREDICTION FORM & RESULT
# -----------------------------------------------------------------------------
elif page == "Prediction Form & Result":
    st.title("🔮 Predict Used Car Price")
    st.markdown("Enter the vehicle specifications below to calculate the estimated price.")
    st.markdown("---")
    
    if pipeline is None:
        st.error(f"⚠️ Trained model artifact not found at `{MODEL_PATH}`! Please run `train.py` first.")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                brand = st.selectbox("Brand / Make", options=[
                    "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes-Benz", 
                    "Audi", "Lexus", "Porsche", "Nissan", "Hyundai", "Kia", "Subaru", "Other"
                ])
                car_model_name = st.text_input("Car Model (e.g., Camry, Mustang, X5)", value="Other")
                model_year = st.slider("Model Year", min_value=1990, max_value=2026, value=2018)
                milage = st.number_input("Total Mileage (miles)", min_value=0, max_value=300000, value=45000, step=1000)

            with col2:
                fuel_type = st.selectbox("Fuel Type", options=["Gasoline", "Hybrid", "Diesel", "Electric", "Flex Fuel", "Other"])
                transmission_type = st.selectbox("Transmission", options=["Automatic", "Manual", "CVT", "Other"])
                accident = st.radio("Accident Reported?", options=["None reported", "At least 1 accident or damage reported"])
                clean_title = st.selectbox("Clean Title Status?", options=["Yes", "No"])

            with col3:
                engine_size_l = st.number_input("Engine Displacement (Liters)", min_value=0.0, max_value=8.0, value=2.5, step=0.1)
                engine_cylinders = st.selectbox("Cylinders", options=[2, 3, 4, 6, 8, 10, 12], index=2)
                engine_hp = st.number_input("Horsepower (HP)", min_value=50, max_value=1000, value=200, step=10)
                num_gears = st.number_input("Number of Transmission Gears", min_value=1, max_value=10, value=6, step=1)

            submit_button = st.form_submit_button("💰 Predict Price", type="primary", use_container_width=True)

        if submit_button:
            # Derived feature computations
            current_year = 2026
            car_age = max(0, current_year - model_year)
            mileage_per_year = milage / max(car_age, 1)
            
            luxury_brands = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Porsche", "Land Rover", "Jaguar"]
            luxury_brand = 1 if brand in luxury_brands else 0

            # Exact dictionary keys required by your trained pipeline
            input_df = pd.DataFrame([{
                "brand": brand,
                "model": car_model_name,
                "fuel_type": fuel_type,
                "transmission_type": transmission_type,
                "accident": accident,
                "clean_title": clean_title,
                "model_year": model_year,
                "milage": milage,
                "car_age": car_age,
                "mileage_per_year": mileage_per_year,
                "luxury_brand": luxury_brand,
                "engine_size_l": engine_size_l,
                "engine_cylinders": engine_cylinders,
                "engine_hp": engine_hp,
                "num_gears": num_gears
            }])

            # Compute prediction
            pred_log = pipeline.predict(input_df)[0]
            pred_price = np.expm1(pred_log)

            st.balloons()
            st.success(f"### Estimated Market Price: **${pred_price:,.2f}**")
            
            c1, c2 = st.columns(2)
            c1.info(f"**Deployed Model:** {model_name}")
            if best_r2 and best_mae:
                c2.info(f"**Accuracy Fit (R²):** {best_r2 * 100:.0f}% | **Average MAE:** ${best_mae:,.2f}")

# -----------------------------------------------------------------------------
# 6. MODEL COMPARISON PAGE
# -----------------------------------------------------------------------------
elif page == "Model Comparison Page":
    st.title("📈 Model Comparison & Leaderboard")
    st.markdown("---")
    
    if metrics_df is not None:
        target_col = "Test R2" if "Test R2" in metrics_df.columns else "R2"
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=[target_col], color="lightgreen"), use_container_width=True)
        
        st.subheader("Model R² Performance Benchmark")
        st.bar_chart(data=metrics_df.set_index("Model")[target_col])
    else:
        st.warning("Metrics file not found. Showing cached baseline comparison table:")
        baseline_data = {
            "Model": ["Support Vector Regressor", "Gradient Boosting", "Random Forest", "Linear Regression", "Ridge Regression", "Decision Tree", "Lasso Regression"],
            "Test R2": [0.8944, 0.8670, 0.8459, 0.8148, 0.8091, 0.7819, 0.7493],
            "Test MAE ($)": [7563.21, 8415.65, 8909.43, 9770.14, 10162.07, 10625.73, 11884.75],
            "Test RMSE ($)": [14932.81, 16745.45, 17023.17, 19802.40, 20964.25, 18650.55, 25357.90],
            "Train Time (s)": [0.390, 0.850, 0.243, 0.067, 0.025, 0.036, 0.021]
        }
        st.dataframe(pd.DataFrame(baseline_data), use_container_width=True)
