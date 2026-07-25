import pandas as pd

def feature_engineering(input_path="/data/used_cars_clean.csv", output_path="/data/used_cars_clean_fe.csv"):
    
    df = pd.read_csv(input_path)
    
    df['car_age'] = 2026 - df["model_year"]