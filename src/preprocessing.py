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
    
    # 7. Save and return
    df.to_csv(output_data, index=False)
    return df
    
if __name__ == "__main__":
    clean()