"""
Week 1 Task: Strategic Planning & Data Exploration in Logistics
Role: Logistics Data Analyst Intern
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def generate_logistics_data():
    np.random.seed(42)
    n_samples = 200

    distances = np.random.uniform(1.0, 25.0, n_samples)
    traffic_levels = np.random.choice(["Low", "Medium", "High"], n_samples)
    weights = np.random.uniform(0.5, 15.0, n_samples)

    traffic_factor = {"Low": 1.1, "Medium": 1.5, "High": 2.2}
    delivery_times = [
        (d * traffic_factor[t]) + (w * 0.4) + np.random.normal(5, 2)
        for d, t, w in zip(distances, traffic_levels, weights)
    ]

    return pd.DataFrame(
        {
            "order_id": range(1001, 1001 + n_samples),
            "distance_km": np.round(distances, 2),
            "traffic_level": traffic_levels,
            "package_weight_kg": np.round(weights, 2),
            "delivery_time_min": np.round(delivery_times, 2),
        }
    )


def run_spatial_clustering():
    coordinates = np.array(
        [
            [28.7041, 77.1025],
            [28.7055, 77.1040],
            [28.5355, 77.3910],
            [28.5360, 77.3925],
            [28.4595, 77.0266],
            [28.4600, 77.0280],
        ]
    )
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clusters = pd.DataFrame(coordinates, columns=["Latitude", "Longitude"])
    df_clusters["Cluster_Zone"] = kmeans.fit_predict(coordinates)
    return df_clusters


def train_predictive_model(df):
    traffic_map = {"Low": 1, "Medium": 2, "High": 3}
    df["traffic_score"] = df["traffic_level"].map(traffic_map)

    X = df[["distance_km", "traffic_score", "package_weight_kg"]]
    y = df["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return r2_score(y_test, preds), mean_absolute_error(y_test, preds)


if __name__ == "__main__":
    data = generate_logistics_data()
    zones = run_spatial_clustering()
    r2, mae = train_predictive_model(data)
    print(f"R2 Score: {r2:.4f}, MAE: {mae:.2f} mins")
