import os
import pandas as pd


def generate_data(csv_path='data/used_cars.csv', output_path='outputs/reports/data_understanding.md'):
    df = pd.read_csv(csv_path)
    
    # Row and Column count
    shape = f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
    
    # Duplicates Count
    duplicates = f"Duplicates: {df.duplicated().sum()}"
    
    # Data Types and Missing Values
    info_table = (
        pd.DataFrame({"Data Types": df.dtypes, "Missing Values": df.isnull().sum()}).to_markdown()
    )
    
    # Summary Statistics
    describe = df.describe(include='all').to_markdown()
    
    report = f"{shape}\n{duplicates}\n{info_table}\n{describe}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
        
if __name__ == '__main__':
    generate_data()