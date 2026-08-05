import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_graphs(df_path, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(df_path)

    # 1. Price & Mileage Distributions
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df["price"], bins=30, kde=True)
    plt.title("Price Distribution")

    plt.subplot(1, 2, 2)
    sns.histplot(df["milage"], bins=30, kde=True)
    plt.title("Mileage Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distributions.png")
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    num_cols = df.select_dtypes(include=["number"])
    sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # 3. Price vs Mileage Scatter
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="milage", y="price", alpha=0.5)
    plt.title("Price vs Mileage")
    plt.savefig(f"{output_dir}/scatter_price_mileage.png")
    plt.close()

    # 4. Top Brands by Count
    plt.figure(figsize=(8, 4))
    top_brands = df["brand"].value_counts().head(10).index
    sns.countplot(
        data=df[df["brand"].isin(top_brands)], y="brand", order=top_brands
    )
    plt.title("Top 10 Brands")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_brands.png")
    plt.close()

    # 5. Price by Accident Status
    if "accident" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="accident", y="price")
        plt.title("Price vs Accident History")
        plt.savefig(f"{output_dir}/accident_vs_price.png")
        plt.close()

    print("Saved plots to", output_dir)
    
def model_comparision(metrics_output):
    # 2. Load the exported metrics summary table
    df = pd.read_csv(metrics_output)

    # 3. Plot Comparison Grid
    os.makedirs("../outputs/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    plots = [
        ("Test R2", "R² Score (Higher is Better)", True),
        ("Test RMSE ($)", "RMSE ($) (Lower is Better)", False),
        ("Test MAE ($)", "MAE ($) (Lower is Better)", False),
        ("Train Time (s)", "Training Time (s) (Lower is Better)", False)
    ]

    for i, (col, title, higher_is_better) in enumerate(plots):
        # Sort models based on metric preference for horizontal bar chart visual clarity
        df_sorted = df.sort_values(by=col, ascending=higher_is_better)
        
        axes[i].barh(df_sorted["Model"], df_sorted[col], color="steelblue")
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("../outputs/figures/comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    generate_graphs("data/used_cars_clean_fe.csv")