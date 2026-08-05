import pandas as pd
import numpy as np

def clean(input_data="../data/used_cars.csv", output_data="../data/used_cars_clean.csv"):
    
    df = pd.read_csv(input_data)
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Trim whitespaces and normalize text
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(["nan", "none", ""], np.nan)
    
    # 3. Clean numeric fields (Price and Mileage)
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    df["milage"] = pd.to_numeric(df["milage"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    
    df.loc[df["price"] <= 0, "price"] = np.nan
    df.loc[df["milage"] < 0, "milage"] = np.nan
    df.loc[df["model_year"] > 2026, 'model_year'] = np.nan
    
    
    price_cap = df["price"].quantile(0.99)
    df.loc[df["price"] > price_cap, "price"] = np.nan

    milage_cap = df["milage"].quantile(0.99)
    df.loc[df["milage"] > milage_cap, "milage"] = np.nan


    df = df.dropna(subset=["price"])
        
    print(f"[INFO] Shape After 1% Outlier Trimming: {df.shape}")
    # 4. Clean binary fields
    df["clean_title"] = df["clean_title"].fillna("unknown")
    df["accident"] = df["accident"].replace({
        "none reported": "no", 
        "at least 1 accident or damage reported": "yes"
    }).fillna("unknown")    
    
    # 6. Fill missing values across categorical columns
    df["fuel_type"] = df["fuel_type"].fillna("unknown")
    
    # 7. Save and return
    df.to_csv(output_data, index=False)
    return df
    
if __name__ == "__main__":
    clean()