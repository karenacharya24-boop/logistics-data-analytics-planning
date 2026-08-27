"""
Week 4 Task: Predictive Modeling and Optimization in Logistics Systems
Role: Logistics Data Analyst Intern
Description: End-to-end predictive modeling pipeline comparing Linear Regression, 
             Random Forest, and Gradient Boosting with Hyperparameter Tuning 
             and prescriptive dispatch optimization.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


def generate_logistics_data():
    """Generates synthetic operational delivery records."""
    np.random.seed(42)
    n_samples = 600

    distance = np.random.uniform(2, 45, n_samples)
    traffic_score = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])
    stops = np.random.randint(1, 8, n_samples)
    weight = np.random.uniform(0.5, 30.0, n_samples)
    weather_delay = np.random.choice([0, 10, 25], n_samples, p=[0.6, 0.3, 0.1])

    # Target: Delivery Time (min)
    delivery_time = (
        (distance * 1.5)
        + (traffic_score * 8.0)
        + (stops * 4.5)
        + (weight * 0.35)
        + weather_delay
        + np.random.normal(0, 4, n_samples)
    )

    df = pd.DataFrame(
        {
            "Distance_km": distance,
            "Traffic_Score": traffic_score,
            "Intermediate_Stops": stops,
            "Package_Weight_kg": weight,
            "Weather_Delay_min": weather_delay,
            "Delivery_Time_min": np.clip(delivery_time, 15, 200),
        }
    )
    return df


def train_and_tune_model(df):
    """Trains regression models, tunes Gradient Boosting, and evaluates test performance."""
    X = df.drop(columns=["Delivery_Time_min"])
    y = df["Delivery_Time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning using 5-Fold Cross-Validation
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="r2",
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("--- Model Performance on Test Set ---")
    print(f"Optimal Parameters: {grid.best_params_}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} mins")
    print(
        f"Root Mean Squared Error (RMSE):"
        f" {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} mins"
    )
    print(f"R-squared (R2) Score: {r2_score(y_test, y_pred):.4f}")

    return best_model, X.columns


def prescriptive_dispatch_engine(
    model,
    feature_cols,
    order_distance,
    traffic_level,
    stops,
    weight,
    weather_flag,
):
    """Prescribes automated logistics actions based on model duration forecasts."""
    order_features = pd.DataFrame(
        [[order_distance, traffic_level, stops, weight, weather_flag]],
        columns=feature_cols,
    )
    predicted_duration = model.predict(order_features)[0]

    # Optimization Logic
    if predicted_duration > 60.0 or traffic_level == 3:
        action = (
            "REROUTE_URBAN_EXPRESS: Split batch & assign lightweight fleet"
        )
    else:
        action = "STANDARD_DISPATCH: Direct single-vehicle fulfillment"

    return predicted_duration, action


if __name__ == "__main__":
    data = generate_logistics_data()
    best_model, cols = train_and_tune_model(data)

    # Test sample dispatch decision
    eta, action = prescriptive_dispatch_engine(
        best_model, cols, order_distance=24.5, traffic_level=3, stops=5, weight=14.0, weather_flag=0
    )
    print("\n--- Prescriptive Optimization Output ---")
    print(f"Sample Route ETA: {eta:.2f} mins")
    print(f"Prescribed Dispatch Action: {action}")
