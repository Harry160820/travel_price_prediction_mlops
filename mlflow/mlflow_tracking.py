"""
MLFlow Tracking for Travel MLOps Capstone Project
Tracks experiments, parameters, metrics, and models
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================

# Set MLFlow tracking URI (change to your MLFlow server URL)
mlflow.set_tracking_uri("http://localhost:5000")  # Or use "file:///path/to/mlruns"

# Set experiment name
EXPERIMENT_NAME = "Flight_Price_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLFlow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment Name: {EXPERIMENT_NAME}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("LOADING AND PREPROCESSING DATA")
print("="*80)

BASE_DIR = Path(__file__).resolve().parents[1]   # project root
DATA_DIR = BASE_DIR / "data"

# Load datasets
users_df = pd.read_csv(DATA_DIR / "users.csv")
flights_df = pd.read_csv(DATA_DIR / "flights.csv")

# Merge datasets
df = flights_df.merge(users_df, left_on='userCode', right_on='code', how='left')

# Date features
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['from', 'to', 'flightType', 'agency', 'gender', 'company']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Derived features
df['price_per_km'] = df['price'] / (df['distance'] + 1)
df['speed'] = df['distance'] / (df['time'] + 1)

# Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])

# Select features
feature_cols = [
    'distance', 'time', 'from_encoded', 'to_encoded',
    'flightType_encoded', 'agency_encoded',
    'day_of_week', 'month', 'is_weekend',
    'age', 'gender_encoded', 'company_encoded',
    'price_per_km', 'speed', 'age_group'
]

# Prepare data
df_clean = df[feature_cols + ['price']].dropna()
X = df_clean[feature_cols]
y = df_clean['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================================
# EXPERIMENT 1: BASELINE MODELS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 1: BASELINE MODELS")
print("="*80)

# Define baseline models
baseline_models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'RandomForest_Basic': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting_Basic': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and log each baseline model
for model_name, model in baseline_models.items():
    with mlflow.start_run(run_name=f"Baseline_{model_name}"):
        
        print(f"\nTraining {model_name}...")
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", len(feature_cols))
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        
        # Log model
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        # Log artifacts
        joblib.dump(model, f'{model_name}_model.pkl')
        mlflow.log_artifact(f'{model_name}_model.pkl')
        
        # Add tags
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("dataset", "flights")
        
        print(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

# ============================================================================
# EXPERIMENT 2: HYPERPARAMETER TUNING FOR RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 2: HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*80)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [15, 25],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

# Different combinations to try
hyperparameter_configs = [
    {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5,  'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 25, 'min_samples_split': 10, 'min_samples_leaf': 4},
    {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 2},
    {'n_estimators': 150, 'max_depth': 25, 'min_samples_split': 5,  'min_samples_leaf': 4},
]

best_model = None
best_r2 = -float('inf')

for idx, params in enumerate(hyperparameter_configs):
    with mlflow.start_run(run_name=f"RandomForest_Tuned_{idx+1}"):
        
        print(f"\nTuning configuration {idx+1}/{len(hyperparameter_configs)}...")
        
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        mlflow.log_param("model_type", "RandomForestRegressor")
        
        # Train model with these parameters
        model = RandomForestRegressor(random_state=42, **params)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("overfit_score", train_r2 - test_r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Add tags
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("tuning_method", "manual")
        
        print(f"  Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
        
        # Track best model
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model = model
            best_params = params

print(f"\nBest model R²: {best_r2:.4f}")
print(f"Best parameters: {best_params}")

# ============================================================================
# EXPERIMENT 3: FINAL PRODUCTION MODEL
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 3: FINAL PRODUCTION MODEL")
print("="*80)

with mlflow.start_run(run_name="Production_Model_v1"):
    
    # Use best parameters found
    production_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Log all parameters
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("num_features", len(feature_cols))
    mlflow.log_param("feature_scaling", "StandardScaler")
    
    # Train model
    print("Training production model...")
    production_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = production_model.predict(X_train_scaled)
    y_pred_test = production_model.predict(X_test_scaled)
    
    # Comprehensive metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Log metrics
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("overfit_score", train_r2 - test_r2)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': production_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log top features as parameters
    for idx, row in feature_importance.head(10).iterrows():
        mlflow.log_param(f"top_feature_{idx+1}", row['feature'])
        mlflow.log_metric(f"importance_{idx+1}", row['importance'])
    
    # Log model with signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train_scaled, y_train)
    
    mlflow.sklearn.log_model(
        production_model,
        "production_model",
        signature=signature,
        registered_model_name="flight_price_predictor"
    )
    
    # Log preprocessing objects
    joblib.dump(scaler, 'production_scaler.pkl')
    joblib.dump(label_encoders, 'production_encoders.pkl')
    joblib.dump(feature_cols, 'production_features.pkl')
    
    mlflow.log_artifact('production_scaler.pkl')
    mlflow.log_artifact('production_encoders.pkl')
    mlflow.log_artifact('production_features.pkl')
    
    # Save feature importance plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'].head(10), 
             feature_importance['importance'].head(10))
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()
    
    # Add comprehensive tags
    mlflow.set_tag("stage", "production")
    mlflow.set_tag("model_version", "v1")
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("deployment_ready", "true")
    mlflow.set_tag("model_purpose", "flight_price_prediction")
    
    print(f"\nProduction Model Metrics:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAE: {test_mae:.2f}")

# ============================================================================
# MODEL REGISTRY
# ============================================================================

print("\n" + "="*80)
print("MODEL REGISTRY")
print("="*80)

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest production model
model_name = "flight_price_predictor"

# Transition model to production stage
try:
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    
    print(f"Model '{model_name}' version {latest_version} transitioned to Production")
    
except Exception as e:
    print(f"Note: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MLFLOW TRACKING SUMMARY")
print("="*80)
print(f"Experiment Name: {EXPERIMENT_NAME}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Total Runs: {len(baseline_models) + len(hyperparameter_configs) + 1}")
print("\nTo view results, run:")
print("  mlflow ui")
print("Then open http://localhost:5000 in your browser")
print("="*80)