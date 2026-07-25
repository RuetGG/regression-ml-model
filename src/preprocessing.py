import pandas as pd
import numpy as np

def clean(input_data="data/used_cars.csv", output_data="data/used_cars_clean.csv"):
    
    df = pd.read_csv(input_data)
    
    # remove duplicates
    df = df.drop_duplicates()
    
    # trim whitespaces and normalize text (to lower and NaN strings are changed to actual nan value)
    str_cols = df.select_dtypes(include=["str"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(["nan", "none", ""], np.nan)
    
    # change price and milage to integer -> ("34,742 mi.") remove mi. ("$54,598") remove $
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce",)
    df["milage"] = pd.to_numeric(df["milage"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    
    # change binary fields such as clean title and accident ->
    # ??? Clean title if it's empty should I consider it as no or should it be Unknown
    df["clean_title"] = df["clean_title"].replace({"yes":"yes", np.nan:"unknown"})
    # ??? And for accident, there are 2 values and should be changed to yes and no
    df["accident"] = df["accident"].replace({"none reported": "no", "at least 1 accident or damage reported": "yes"})

    df["accident"] = df["accident"].fillna("unknown")
    
    # check missing values -> fuel_type "Unknown", accident "Unknown", clean_title "Unknown"
    df["fuel_type"] = df["fuel_type"].fillna("Unknown")
    
    df.to_csv(output_data, index=False)
    return df

clean()

  