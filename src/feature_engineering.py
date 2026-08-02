import pandas as pd
import numpy as np

def feature_engineering(input_path="data/used_cars_clean.csv", output_path="data/used_cars_clean_fe.csv"):
    
    df = pd.read_csv(input_path)
    df = df.copy()
    
    # Car age 
    df['car_age'] = 2026 - df["model_year"]
    
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