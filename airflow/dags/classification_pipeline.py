"""
Airflow DAG for Gender Classification Model Pipeline
Handles data loading, preprocessing, training, evaluation, and model deployment
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, classification_report)
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
    'gender_classification_pipeline',
    default_args=default_args,
    description='Complete pipeline for gender classification model',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    tags=['ml', 'classification', 'gender'],
)

# ============================================================================
# TASK 1: DATA EXTRACTION
# ============================================================================

def extract_data(**kwargs):
    """Load datasets from source"""
    print("Starting data extraction...")
    
    data_path = '/opt/airflow/data/'
    
    users_df = pd.read_csv(f'{data_path}users.csv')
    flights_df = pd.read_csv(f'{data_path}flights.csv')
    hotels_df = pd.read_csv(f'{data_path}hotels.csv')
    
    print(f"Users data shape: {users_df.shape}")
    print(f"Flights data shape: {flights_df.shape}")
    print(f"Hotels data shape: {hotels_df.shape}")
    
    # Store in XCom
    kwargs['ti'].xcom_push(key='users_data', value=users_df.to_json())
    kwargs['ti'].xcom_push(key='flights_data', value=flights_df.to_json())
    kwargs['ti'].xcom_push(key='hotels_data', value=hotels_df.to_json())
    
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
    hotels_json = ti.xcom_pull(key='hotels_data', task_ids='extract_data')
    
    users_df = pd.read_json(users_json)
    flights_df = pd.read_json(flights_json)
    hotels_df = pd.read_json(hotels_json)
    
    # Validation checks
    assert not users_df.empty, "Users data is empty"
    assert 'gender' in users_df.columns, "Gender column missing"
    assert users_df['gender'].notna().sum() > 0, "No valid gender labels"
    
    # Check class distribution
    gender_counts = users_df['gender'].value_counts()
    print(f"Gender distribution: {gender_counts.to_dict()}")
    
    # Check for severely imbalanced classes
    min_class_count = gender_counts.min()
    max_class_count = gender_counts.max()
    imbalance_ratio = max_class_count / min_class_count
    
    if imbalance_ratio > 10:
        print(f"Warning: Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
    
    print(f"Validation passed! Total users: {len(users_df)}")
    
    # Pass data to next task
    kwargs['ti'].xcom_push(key='validated_users', value=users_json)
    kwargs['ti'].xcom_push(key='validated_flights', value=flights_json)
    kwargs['ti'].xcom_push(key='validated_hotels', value=hotels_json)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# ============================================================================
# TASK 3: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(**kwargs):
    """Create features for classification"""
    print("Starting feature engineering...")
    
    ti = kwargs['ti']
    users_json = ti.xcom_pull(key='validated_users', task_ids='validate_data')
    flights_json = ti.xcom_pull(key='validated_flights', task_ids='validate_data')
    hotels_json = ti.xcom_pull(key='validated_hotels', task_ids='validate_data')
    
    users_df = pd.read_json(users_json)
    flights_df = pd.read_json(flights_json)
    hotels_df = pd.read_json(hotels_json)
    
    # Aggregate flight features per user
    flight_features = flights_df.groupby('userCode').agg({
        'price': ['mean', 'sum', 'count', 'std'],
        'time': ['mean', 'sum'],
        'distance': ['mean', 'sum', 'max'],
    }).reset_index()
    
    flight_features.columns = ['userCode', 'avg_flight_price', 'total_flight_spent', 
                              'num_flights', 'std_flight_price',
                              'avg_flight_time', 'total_flight_time',
                              'avg_distance', 'total_distance', 'max_distance']
    
    # Aggregate hotel features per user
    hotel_features = hotels_df.groupby('userCode').agg({
        'price': ['mean', 'sum'],
        'days': ['mean', 'sum'],
        'total': ['mean', 'sum', 'count']
    }).reset_index()
    
    hotel_features.columns = ['userCode', 'avg_hotel_price', 'total_hotel_price',
                             'avg_hotel_days', 'total_hotel_days',
                             'avg_hotel_total', 'total_hotel_spent', 'num_hotel_bookings']
    
    # Flight type preferences
    flight_type_pivot = pd.crosstab(flights_df['userCode'], flights_df['flightType'])
    flight_type_pivot = flight_type_pivot.add_prefix('flightType_')
    flight_type_pivot = flight_type_pivot.reset_index()
    
    # Merge all features with users
    df = users_df.copy()
    df = df.merge(flight_features, left_on='code', right_on='userCode', how='left')
    df = df.merge(hotel_features, left_on='code', right_on='userCode', how='left')
    df = df.merge(flight_type_pivot, left_on='code', right_on='userCode', how='left')
    
    # Fill missing values
    df = df.fillna(0)
    
    # Create additional features
    df['total_travel_spent'] = df['total_flight_spent'] + df['total_hotel_spent']
    df['avg_trip_value'] = df['total_travel_spent'] / (df['num_flights'] + 1)
    df['flight_to_hotel_ratio'] = df['num_flights'] / (df['num_hotel_bookings'] + 1)
    df['avg_stay_length'] = df['total_hotel_days'] / (df['num_hotel_bookings'] + 1)
    
    # Encode company
    le_company = LabelEncoder()
    df['company_encoded'] = le_company.fit_transform(df['company'].astype(str))
    
    print(f"Features created! Total columns: {len(df.columns)}")
    print(f"Total samples: {len(df)}")
    
    # Save engineered data and encoders
    kwargs['ti'].xcom_push(key='engineered_data', value=df.to_json())
    
    # Save label encoder
    output_path = '/opt/airflow/models/'
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(le_company, f'{output_path}company_encoder.pkl')
    
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
    """Train the classification model"""
    print("Starting model training...")
    
    ti = kwargs['ti']
    data_json = ti.xcom_pull(key='engineered_data', task_ids='feature_engineering')
    df = pd.read_json(data_json)
    
    # Select features
    feature_cols = [
        'age', 'avg_flight_price', 'total_flight_spent', 'num_flights', 
        'std_flight_price', 'avg_flight_time', 'total_flight_time',
        'avg_distance', 'total_distance', 'max_distance',
        'avg_hotel_price', 'total_hotel_price', 'avg_hotel_days', 
        'total_hotel_days', 'num_hotel_bookings',
        'total_travel_spent', 'avg_trip_value', 'flight_to_hotel_ratio',
        'avg_stay_length', 'company_encoded'
    ]
    
    # Add flight type columns
    flight_type_cols = [col for col in df.columns if col.startswith('flightType_')]
    feature_cols.extend(flight_type_cols)
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['gender']
    
    # Encode target variable
    le_gender = LabelEncoder()
    y_encoded = le_gender.fit_transform(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y_encoded)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf_model, param_grid, cv=3, 
        scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    
    print("Training Random Forest with Grid Search...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    model = grid_search.best_estimator_
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
        'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
        'test_f1': f1_score(y_test, y_pred_test, average='weighted')
    }
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=le_gender.classes_))
    
    # Save model and preprocessors
    output_path = '/opt/airflow/models/'
    joblib.dump(model, f'{output_path}gender_classifier.pkl')
    joblib.dump(scaler, f'{output_path}gender_scaler.pkl')
    joblib.dump(le_gender, f'{output_path}gender_label_encoder.pkl')
    joblib.dump(feature_cols, f'{output_path}gender_feature_columns.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'Random Forest Classifier',
        'best_params': grid_search.best_params_,
        'test_accuracy': metrics['test_accuracy'],
        'test_f1': metrics['test_f1'],
        'test_precision': metrics['test_precision'],
        'test_recall': metrics['test_recall'],
        'classes': le_gender.classes_.tolist(),
        'features': feature_cols,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_size': len(df)
    }
    joblib.dump(metadata, f'{output_path}gender_model_metadata.pkl')
    
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
    """Evaluate model performance"""
    print("Starting model evaluation...")
    
    ti = kwargs['ti']
    metrics = ti.xcom_pull(key='model_metrics', task_ids='train_model')
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50 + "\n")
    
    # Quality threshold
    if metrics['test_accuracy'] < 0.70:
        print(
            f"WARNING: Accuracy below threshold "
            f"({metrics['test_accuracy']:.4f} < 0.70)"
        )

    if metrics['test_f1'] < 0.70:
        print(
            f"WARNING: F1 below threshold "
            f"({metrics['test_f1']:.4f} < 0.70)"
        )
    
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
    echo "Deploying gender classification model to production..."
    
    # Copy model files to deployment directory
    cp /opt/airflow/models/gender_*.pkl /opt/deployment/models/
    
    # Restart API service
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
    Gender Classification Model Pipeline Completed Successfully!
    
    Metrics:
    - Test Accuracy: {metrics['test_accuracy']:.4f}
    - Test F1 Score: {metrics['test_f1']:.4f}
    - Test Precision: {metrics['test_precision']:.4f}
    - Test Recall: {metrics['test_recall']:.4f}
    
    Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    print(message)
    # In production: send email or Slack notification

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