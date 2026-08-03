import pandas as pd
import numpy as np

def clean(input_data="data/used_cars.csv", output_data="data/used_cars_clean.csv"):
    
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
    
    # 4. Clean binary fields
    df["clean_title"] = df["clean_title"].fillna("unknown")
    df["accident"] = df["accident"].replace({
        "none reported": "no", 
        "at least 1 accident or damage reported": "yes"
    }).fillna("unknown")
    
    # 5. Clean transmission column & engineer new features using .apply()
    trans_str = df["transmission"].astype(str).str.lower()
    
    # Extract numerical gear counts (e.g., 6, 8, 10) into a separate column
    gears_extracted = trans_str.str.extract(r'(\d+)')[0] 
    df['num_gears'] = gears_extracted.fillna('unknown')
    
    # Helper function for row-by-row mapping
    def map_transmission(text):
        if pd.isna(text) or text == 'nan':
            return np.nan
        
        # Check for junk text first
        if 'scheduled' in text or 'production' in text:
            return np.nan
        
        # Priority mapping
        if any(term in text for term in ['m/t', 'manual', ' mt ']) or text.endswith(' mt'):
            return 'manual'
        elif 'cvt' in text or 'variable' in text:
            return 'cvt'
        elif any(term in text for term in ['a/t', 'auto', 'shift mode', 'dct']):
            return 'automatic'
        else:
            return np.nan  # Returns NaN for unmapped/unknown values

    # Apply the mapping function
    df["transmission_type"] = trans_str.apply(map_transmission)
    
    # 6. Fill missing values across categorical columns
    df["fuel_type"] = df["fuel_type"].fillna("unknown")
    
    # Updated - DYNAMIC STATISTICAL OUTLIER DETECTION 
    # 1. Nullify physically impossible errors
    if "price" in df.columns:
        df.loc[df["price"] <= 0, "price"] = np.nan
    
    mileage_col = "mileage" if "mileage" in df.columns else "milage"
    if mileage_col in df.columns:
        df.loc[df[mileage_col] < 0, mileage_col] = np.nan
        
    if "model_year" in df.columns:
        current_year = pd.Timestamp.now().year + 1
        df.loc[df["model_year"] > current_year, "model_year"] = np.nan

    # 2. Dynamic percentile-based outlier detection (top 1% capped to NaN)
    for col in ["price", mileage_col]:
        if col in df.columns and df[col].notna().any():
            upper_limit = df[col].quantile(0.99)
            df.loc[df[col] > upper_limit, col] = np.nan

    # Updated - STANDARD SEQUENTIAL IMPUTATION 
    # 1. Drop price rows with missing values (including outliers set to NaN)
    if "price" in df.columns:
        df = df.dropna(subset=["price"])

    # 2. Numeric features: Impute missing values with MEDIAN
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col != "price" and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # 3. Categorical features: Impute missing values with MODE (or 'unknown')
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "unknown"
            df[col] = df[col].fillna(fill_val)

    # 7. Save and return
    df.to_csv(output_data, index=False)
    return df
    
if __name__ == "__main__":
    clean()