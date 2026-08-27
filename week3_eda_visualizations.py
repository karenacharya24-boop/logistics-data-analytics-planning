"""
Week 3 Task: Advanced Data Analysis and Visualization in Logistics
Role: Logistics Data Analyst Intern
Description: Python script for statistical EDA, correlation matrices,
             and multi-panel visualizations using Matplotlib and Seaborn.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_logistics_eda_dataset():
    """Generates logistics dataset with multivariate operational features."""
    np.random.seed(42)
    n_samples = 400

    distances = np.random.uniform(3, 40, n_samples)
    traffic_score = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])
    vehicle_types = np.random.choice(
        ["Electric Bike", "Transit Van", "Heavy Truck"],
        n_samples,
        p=[0.35, 0.45, 0.20],
    )
    weights = np.random.uniform(1.0, 30.0, n_samples)

    speed_factor = {
        "Electric Bike": 1.8,
        "Transit Van": 1.2,
        "Heavy Truck": 1.5,
    }
    traffic_mult = {1: 1.0, 2: 1.35, 3: 1.8}
    base_times = [
        d * speed_factor[v] * traffic_mult[t] + (w * 0.3) + np.random.normal(5, 3)
        for d, v, t, w in zip(distances, vehicle_types, traffic_score, weights)
    ]

    cost_per_km = {
        "Electric Bike": 0.8,
        "Transit Van": 1.9,
        "Heavy Truck": 3.5,
    }
    transport_costs = [
        15.0 + (d * cost_per_km[v]) + (t * 4.5) + np.random.normal(3, 1.5)
        for d, v, t in zip(distances, vehicle_types, traffic_score)
    ]

    return pd.DataFrame(
        {
            "Distance_km": distances,
            "Traffic_Congestion_Score": traffic_score,
            "Package_Weight_kg": weights,
            "Vehicle_Type": vehicle_types,
            "Delivery_Time_min": np.clip(base_times, 10, 180),
            "Transport_Cost_USD": np.clip(transport_costs, 10, 200),
        }
    )


def generate_visualizations(df):
    """Builds and exports operational correlation heatmaps and boxplots."""
    # 1. Correlation Heatmap
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="white")
    corr = df[
        [
            "Distance_km",
            "Traffic_Congestion_Score",
            "Package_Weight_kg",
            "Delivery_Time_min",
            "Transport_Cost_USD",
        ]
    ].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5
    )
    plt.title(
        "Logistics Operational Correlation Matrix", fontweight="bold", pad=10
    )
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.close()

    # 2. Comparative Fleet Boxplot & Scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.boxplot(
        data=df,
        x="Vehicle_Type",
        y="Transport_Cost_USD",
        palette="Blues",
        ax=ax1,
    )
    ax1.set_title(
        "Transportation Cost by Fleet Type", fontweight="bold", fontsize=10
    )
    ax1.set_xlabel("Fleet Category")
    ax1.set_ylabel("Total Cost ($ USD)")

    sns.scatterplot(
        data=df,
        x="Distance_km",
        y="Delivery_Time_min",
        hue="Vehicle_Type",
        palette="Set1",
        alpha=0.8,
        ax=ax2,
    )
    ax2.set_title(
        "Delivery Duration vs Transit Distance", fontweight="bold", fontsize=10
    )
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Delivery Time (Minutes)")
    plt.tight_layout()
    plt.savefig("fleet_cost_time_analysis.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    df = generate_logistics_eda_dataset()
    print("--- Dataset Statistical Summary ---")
    print(df.describe())
    generate_visualizations(df)
    print("\nVisualizations successfully generated and saved.")
