import pandas as pd
import numpy as np

def feature_engineering(input_path="../data/used_cars_clean.csv", output_path="../data/used_cars_clean_fe.csv"):
    
    df = pd.read_csv(input_path)
    df = df.copy()
    
    # Car age 
    df['car_age'] = 2026 - df["model_year"]
    
    if 'milage' in df.columns:
        age_denominator = df["car_age"].clip(lower=1)
        df["milage_per_year"] = df['milage'] / age_denominator
    
    # Luxury Brands 
    luxury_brands = {
        "porsche",
        "ferrari",
        "lamborghini",
        "bentley",
        "rolls-royce",
        "aston martin",
        "maserati",
        "bmw",
        "mercedes-benz",
        "audi",
        "lexus",
        "land rover",
        "jaguar",
        "cadillac",
        "lincoln",
        "genesis",
        "tesla",
        "mclaren",
        "bugatti",
    }
    
    df['luxury_brand'] = (df['brand'].isin(luxury_brands).astype(int))
    
    trans_str = df["transmission"].astype(str).str.lower()
    # Extract numerical gear counts (e.g., 6, 8, 10) into a separate column
    gears_extracted = trans_str.str.extract(r'(\d+)')[0]
    df['num_gears'] = pd.to_numeric(gears_extracted, errors='coerce') 
    
    
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
    
    # Engine parsing (HP, Engine Size, Cylinders)
    engine_str = df["engine"].astype(str).str.lower()

    # Extract Horsepower (e.g., '300.0hp' -> 300.0)
    df["engine_hp"] = (
        engine_str.str.extract(r"\b(\d{2,4}(?:\.\d+)?)\s*hp")[0]
        .astype(float)
        .fillna(np.nan)
    )

    # 2. Extract Engine Size (Liters)
    # Looks for a number like 2.0, 3.7, 5.3 followed by 'l' or 'liter'
    df["engine_size_l"] = (
        engine_str.str.extract(r"\b(\d\.\d+)\s*(?:l|liter)\b")[0]
        .astype(float)
        .fillna(np.nan)
    )

    # 3. Extract Cylinders
    # Looks for 'v6', 'v8', 'i4', '4 cylinder', '8 cylinder', 'straight 6', etc.
    df["engine_cylinders"] = (
        engine_str.str.extract(
            r"(?:v|i|straight\s*|\b)(\d+)\s*(?:cyl|cylinder|\b)"
        )[0]
        .astype(float)
        .fillna(np.nan)
    )

    # 4. Extract Turbo Flag
    df["is_turbo"] = engine_str.str.contains("turbo", na=False).astype(int)
    
    df.to_csv(output_path, index=False)
    print("Feature Engineering Done")

    return df


if __name__ == "__main__":
    feature_engineering()