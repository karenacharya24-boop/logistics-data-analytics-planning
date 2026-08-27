"""
Week 2 Task: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Role: Logistics Data Analyst Intern
Description: Robust pipeline for data cleaning, IQR outlier clipping,
             missing value imputation, and feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def simulate_raw_logistics_data():
    """Generates synthetic telemetry data containing missing values and anomalies."""
    np.random.seed(42)
    n_records = 300

    data = {
        "shipment_id": [f"SHP-{i}" for i in range(1000, 1000 + n_records)],
        "transit_distance_km": np.random.uniform(2.0, 45.0, n_records),
        "package_weight_kg": np.random.uniform(0.5, 25.0, n_records),
        "traffic_density": np.random.choice(
            ["Low", "Medium", "High", np.nan], n_records, p=[0.3, 0.4, 0.2, 0.1]
        ),
        "vehicle_type": np.random.choice(["Van", "Bike", "Truck"], n_records),
        "delivery_time_min": np.random.uniform(15.0, 120.0, n_records),
    }

    df = pd.DataFrame(data)

    # Inject deliberate real-world data quality flaws
    df.loc[np.random.choice(df.index, 15), "package_weight_kg"] = np.nan
    df.loc[np.random.choice(df.index, 5), "delivery_time_min"] = (
        999.0  # Sensor spikes
    )
    df.loc[np.random.choice(df.index, 3), "transit_distance_km"] = (
        -5.0  # Invalid distance entries
    )

    return df


def clean_and_impute(df):
    """Filters invalid records, imputes missing values, and clips outliers."""
    # 1. Filter out impossible domain values
    df_clean = df[df["transit_distance_km"] > 0].copy()

    # 2. Impute continuous data with Median, categorical data with Mode
    df_clean["package_weight_kg"].fillna(
        df_clean["package_weight_kg"].median(), inplace=True
    )
    df_clean["traffic_density"].fillna(
        df_clean["traffic_density"].mode()[0], inplace=True
    )

    # 3. Handle outliers using Interquartile Range (IQR) clipping
    q1 = df_clean["delivery_time_min"].quantile(0.25)
    q3 = df_clean["delivery_time_min"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df_clean["delivery_time_min"] = np.clip(
        df_clean["delivery_time_min"], lower, upper
    )

    return df_clean


def transform_and_normalize(df):
    """Encodes categorical fields and normalizes numerical features to [0, 1]."""
    # Ordinal mapping for traffic density
    traffic_order = {"Low": 1, "Medium": 2, "High": 3}
    df["traffic_score"] = df["traffic_density"].map(traffic_order)

    # One-hot encoding for categorical vehicle types
    df = pd.get_dummies(df, columns=["vehicle_type"], drop_first=False)

    # Min-Max Normalization
    scaler = MinMaxScaler()
    num_cols = ["transit_distance_km", "package_weight_kg", "traffic_score"]
    df[[f"{c}_scaled" for c in num_cols]] = scaler.fit_transform(df[num_cols])

    return df


if __name__ == "__main__":
    raw_df = simulate_raw_logistics_data()
    print("--- 1. Raw Data Shape & Missing Records ---")
    print(raw_df.isnull().sum())

    cleaned_df = clean_and_impute(raw_df)
    final_df = transform_and_normalize(cleaned_df)

    print("\n--- 2. Preprocessed Data Head ---")
    print(final_df.head())
