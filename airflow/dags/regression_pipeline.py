"""
Airflow DAG for Flight Price Regression Model Pipeline
Handles data loading, preprocessing, training, evaluation, and model deployment
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Define the DAG
dag = DAG(
    'flight_price_regression_pipeline',
    default_args=default_args,
    description='Complete pipeline for flight price prediction model',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['ml', 'regression', 'flight-price'],
)

# ============================================================================
# TASK 1: DATA EXTRACTION
# ============================================================================

def extract_data(**kwargs):
    """Load datasets from source"""
    print("Starting data extraction...")
    
    # In production, this would read from database or cloud storage
    data_path = '/opt/airflow/data/'
    
    users_df = pd.read_csv(f'{data_path}users.csv')
    flights_df = pd.read_csv(f'{data_path}flights.csv')
    
    print(f"Users data shape: {users_df.shape}")
    print(f"Flights data shape: {flights_df.shape}")
    
    # Store in XCom for next task
    kwargs['ti'].xcom_push(key='users_data', value=users_df.to_json())
    kwargs['ti'].xcom_push(key='flights_data', value=flights_df.to_json())
    
    print("Data extraction completed!")

extract_data_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

# ============================================================================
# TASK 2: DATA VALIDATION
# ============================================================================

def validate_data(**kwargs):
    """Validate data quality and schema"""
    print("Starting data validation...")
    
    ti = kwargs['ti']
    users_json = ti.xcom_pull(key='users_data', task_ids='extract_data')
    flights_json = ti.xcom_pull(key='flights_data', task_ids='extract_data')
    
    users_df = pd.read_json(users_json)
    flights_df = pd.read_json(flights_json)
    
    # Validation checks
    assert not users_df.empty, "Users data is empty"
    assert not flights_df.empty, "Flights data is empty"
    assert 'price' in flights_df.columns, "Price column missing"
    assert flights_df['price'].notna().sum() > 0, "No valid prices"
    
    # Check for data quality issues
    missing_prices = flights_df['price'].isna().sum()
    if missing_prices > 0:
        print(f"Warning: {missing_prices} missing prices found")
    
    print(f"Validation passed! Records: {len(flights_df)}")
    
    # Pass data to next task
    kwargs['ti'].xcom_push(key='validated_users', value=users_json)
    kwargs['ti'].xcom_push(key='validated_flights', value=flights_json)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# ============================================================================
# TASK 3: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(**kwargs):
    """Create features for modeling"""
    print("Starting feature engineering...")
    
    ti = kwargs['ti']
    users_json = ti.xcom_pull(key='validated_users', task_ids='validate_data')
    flights_json = ti.xcom_pull(key='validated_flights', task_ids='validate_data')
    
    users_df = pd.read_json(users_json)
    flights_df = pd.read_json(flights_json)
    
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
    df['price_per_hour'] = df['price'] / (df['time'] + 1)
    df['speed'] = df['distance'] / (df['time'] + 1)
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                             labels=[0, 1, 2, 3])
    
    print(f"Features created! Total columns: {len(df.columns)}")
    
    # Save engineered data and encoders
    kwargs['ti'].xcom_push(key='engineered_data', value=df.to_json())
    
    # Save label encoders
    output_path = '/opt/airflow/models/'
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(label_encoders, f'{output_path}label_encoders.pkl')
    
    print("Feature engineering completed!")

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

# ============================================================================
# TASK 4: TRAIN MODEL
# ============================================================================

def train_model(**kwargs):
    """Train the regression model"""
    print("Starting model training...")
    
    ti = kwargs['ti']
    data_json = ti.xcom_pull(key='engineered_data', task_ids='feature_engineering')
    df = pd.read_json(data_json)
    
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=25,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    print(f"Training completed!")
    print(f"Train R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    
    # Save model and scaler
    output_path = '/opt/airflow/models/'
    joblib.dump(model, f'{output_path}flight_price_model.pkl')
    joblib.dump(scaler, f'{output_path}scaler.pkl')
    joblib.dump(feature_cols, f'{output_path}feature_columns.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'Random Forest Regressor',
        'metrics': metrics,
        'features': feature_cols,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_size': len(df_clean)
    }
    joblib.dump(metadata, f'{output_path}model_metadata.pkl')
    
    # Push metrics to XCom
    kwargs['ti'].xcom_push(key='model_metrics', value=metrics)
    
    print("Model saved successfully!")

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# ============================================================================
# TASK 5: EVALUATE MODEL
# ============================================================================

def evaluate_model(**kwargs):
    """Evaluate model performance and log metrics"""
    print("Starting model evaluation...")
    
    ti = kwargs['ti']
    metrics = ti.xcom_pull(key='model_metrics', task_ids='train_model')
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50 + "\n")
    
    # Check if model meets quality threshold
    if metrics['test_r2'] < 0.7:
        raise ValueError(f"Model R² ({metrics['test_r2']:.4f}) below threshold (0.7)")
    
    print("Model evaluation passed!")

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# ============================================================================
# TASK 6: DEPLOY MODEL
# ============================================================================

deploy_model_task = BashOperator(
    task_id='deploy_model',
    bash_command="""
    echo "Deploying model to production..."
    # Copy model files to deployment directory
    cp /opt/airflow/models/*.pkl /opt/deployment/models/
    
    # Restart API service (adjust based on your deployment)
    # kubectl rollout restart deployment/travel-ml-api
    
    echo "Model deployment completed!"
    """,
    dag=dag,
)

# ============================================================================
# TASK 7: SEND NOTIFICATION
# ============================================================================

def send_notification(**kwargs):
    """Send notification about pipeline completion"""
    ti = kwargs['ti']
    metrics = ti.xcom_pull(key='model_metrics', task_ids='train_model')
    
    message = f"""
    Flight Price Prediction Model Pipeline Completed Successfully!
    
    Metrics:
    - Test R²: {metrics['test_r2']:.4f}
    - Test RMSE: {metrics['test_rmse']:.2f}
    - Test MAE: {metrics['test_mae']:.2f}
    
    Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    print(message)
    # In production, send email or Slack notification
    # send_slack_notification(message)

send_notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
)

# ============================================================================
# DEFINE TASK DEPENDENCIES
# ============================================================================

extract_data_task >> validate_data_task >> feature_engineering_task
feature_engineering_task >> train_model_task >> evaluate_model_task
evaluate_model_task >> deploy_model_task >> send_notification_task