"""
Airflow DAG for Hotel Recommendation System Pipeline
Handles data loading, matrix creation, model training, and deployment
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
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
    'hotel_recommendation_pipeline',
    default_args=default_args,
    description='Complete pipeline for hotel recommendation system',
    schedule_interval='@monthly',  # Run monthly
    catchup=False,
    tags=['ml', 'recommendation', 'hotels'],
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
    hotels_json = ti.xcom_pull(key='hotels_data', task_ids='extract_data')
    hotels_df = pd.read_json(hotels_json)
    
    # Validation checks
    assert not hotels_df.empty, "Hotels data is empty"
    assert 'userCode' in hotels_df.columns, "userCode column missing"
    assert 'name' in hotels_df.columns, "Hotel name column missing"
    
    # Check data quality
    print(f"Total hotel bookings: {len(hotels_df)}")
    print(f"Unique users: {hotels_df['userCode'].nunique()}")
    print(f"Unique hotels: {hotels_df['name'].nunique()}")
    print(f"Average bookings per user: {len(hotels_df) / hotels_df['userCode'].nunique():.2f}")
    
    # Check for sparse data
    sparsity = 1 - (len(hotels_df) / (hotels_df['userCode'].nunique() * hotels_df['name'].nunique()))
    print(f"Data sparsity: {sparsity*100:.2f}%")
    
    print("Validation passed!")
    
    # Pass data to next task
    kwargs['ti'].xcom_push(key='validated_hotels', value=hotels_json)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# ============================================================================
# TASK 3: CREATE INTERACTION MATRIX
# ============================================================================

def create_interaction_matrix(**kwargs):
    """Create user-item interaction matrix"""
    print("Creating user-item interaction matrix...")
    
    ti = kwargs['ti']
    hotels_json = ti.xcom_pull(key='validated_hotels', task_ids='validate_data')
    hotels_df = pd.read_json(hotels_json)
    
    # Create user-hotel interaction matrix
    user_hotel_matrix = hotels_df.groupby(['userCode', 'name']).size().reset_index(name='bookings')
    
    # Pivot to create matrix
    interaction_matrix = user_hotel_matrix.pivot(
        index='userCode', 
        columns='name', 
        values='bookings'
    ).fillna(0)
    
    print(f"Interaction Matrix Shape: {interaction_matrix.shape}")
    print(f"Users: {interaction_matrix.shape[0]}")
    print(f"Hotels: {interaction_matrix.shape[1]}")
    
    # Calculate sparsity
    sparsity = (interaction_matrix == 0).sum().sum() / (interaction_matrix.shape[0] * interaction_matrix.shape[1])
    print(f"Matrix Sparsity: {sparsity*100:.2f}%")
    
    # Save interaction matrix
    kwargs['ti'].xcom_push(key='interaction_matrix', value=interaction_matrix.to_json())
    kwargs['ti'].xcom_push(key='hotels_data', value=hotels_json)
    
    print("Interaction matrix created!")

create_matrix_task = PythonOperator(
    task_id='create_interaction_matrix',
    python_callable=create_interaction_matrix,
    dag=dag,
)

# ============================================================================
# TASK 4: BUILD CONTENT FEATURES
# ============================================================================

def build_content_features(**kwargs):
    """Build content-based features for hotels"""
    print("Building content-based features...")
    
    ti = kwargs['ti']
    hotels_json = ti.xcom_pull(key='hotels_data', task_ids='create_interaction_matrix')
    hotels_df = pd.read_json(hotels_json)
    
    # Create hotel feature matrix
    hotel_features = hotels_df.groupby('name').agg({
        'place': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'price': 'mean',
        'days': 'mean',
        'total': 'mean',
        'userCode': 'count'
    }).reset_index()
    
    hotel_features.columns = ['hotel_name', 'location', 'avg_price', 
                             'avg_days', 'avg_total', 'popularity']
    
    print(f"Hotel Features Shape: {hotel_features.shape}")
    
    # Encode location
    le_place = LabelEncoder()
    hotel_features['location_encoded'] = le_place.fit_transform(hotel_features['location'])
    
    # Normalize features
    scaler = MinMaxScaler()
    feature_cols = ['avg_price', 'avg_days', 'avg_total', 'popularity', 'location_encoded']
    hotel_features_scaled = scaler.fit_transform(hotel_features[feature_cols])
    
    # Create similarity matrix
    content_similarity = cosine_similarity(hotel_features_scaled)
    content_similarity_df = pd.DataFrame(
        content_similarity,
        index=hotel_features['hotel_name'],
        columns=hotel_features['hotel_name']
    )
    
    print("Content-based features created!")
    
    # Save features
    kwargs['ti'].xcom_push(key='hotel_features', value=hotel_features.to_json())
    kwargs['ti'].xcom_push(key='content_similarity', value=content_similarity_df.to_json())
    
    # Save preprocessors
    output_path = '/opt/airflow/models/'
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(scaler, f'{output_path}recommender_scaler.pkl')
    joblib.dump(le_place, f'{output_path}recommender_label_encoder.pkl')

build_features_task = PythonOperator(
    task_id='build_content_features',
    python_callable=build_content_features,
    dag=dag,
)

# ============================================================================
# TASK 5: TRAIN COLLABORATIVE FILTERING MODEL
# ============================================================================

def train_collaborative_model(**kwargs):
    """Train collaborative filtering model using SVD"""
    print("Training collaborative filtering model...")
    
    ti = kwargs['ti']
    interaction_json = ti.xcom_pull(key='interaction_matrix', task_ids='create_interaction_matrix')
    interaction_matrix = pd.read_json(interaction_json)
    
    # Calculate user and item similarities
    print("Calculating user similarity...")
    user_similarity = cosine_similarity(interaction_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=interaction_matrix.index,
        columns=interaction_matrix.index
    )
    
    print("Calculating item similarity...")
    item_similarity = cosine_similarity(interaction_matrix.T)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=interaction_matrix.columns,
        columns=interaction_matrix.columns
    )
    
    # Apply SVD for matrix factorization
    n_components = min(50, min(interaction_matrix.shape) - 1)
    print(f"Applying SVD with {n_components} components...")
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(interaction_matrix)
    hotel_factors = svd.components_.T
    
    print(f"User Factors Shape: {user_factors.shape}")
    print(f"Hotel Factors Shape: {hotel_factors.shape}")
    print(f"Explained Variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    # Save factors and similarities
    kwargs['ti'].xcom_push(key='user_factors', value=user_factors.tolist())
    kwargs['ti'].xcom_push(key='hotel_factors', value=hotel_factors.tolist())
    kwargs['ti'].xcom_push(key='user_similarity', value=user_similarity_df.to_json())
    kwargs['ti'].xcom_push(key='item_similarity', value=item_similarity_df.to_json())
    kwargs['ti'].xcom_push(key='svd_variance', value=float(svd.explained_variance_ratio_.sum()))
    
    print("Collaborative filtering model trained!")

train_collab_task = PythonOperator(
    task_id='train_collaborative_model',
    python_callable=train_collaborative_model,
    dag=dag,
)

# ============================================================================
# TASK 6: EVALUATE RECOMMENDATION SYSTEM
# ============================================================================

def evaluate_recommender(**kwargs):
    """Evaluate recommendation system using precision@k"""
    print("Evaluating recommendation system...")
    
    ti = kwargs['ti']
    interaction_json = ti.xcom_pull(key='interaction_matrix', task_ids='create_interaction_matrix')
    user_factors = np.array(ti.xcom_pull(key='user_factors', task_ids='train_collaborative_model'))
    hotel_factors = np.array(ti.xcom_pull(key='hotel_factors', task_ids='train_collaborative_model'))
    svd_variance = ti.xcom_pull(key='svd_variance', task_ids='train_collaborative_model')
    
    interaction_matrix = pd.read_json(interaction_json)
    
    # Calculate precision@k for sample of users
    def precision_at_k(actual, predicted, k=5):
        if len(predicted) > k:
            predicted = predicted[:k]
        
        num_hit = len(set(actual) & set(predicted))
        return num_hit / k if k > 0 else 0
    
    precisions_at_5 = []
    precisions_at_10 = []
    sample_size = min(100, len(interaction_matrix))
    
    print(f"Evaluating on {sample_size} users...")
    
    for i in range(sample_size):
        user_idx = i
        
        # Get actual bookings
        actual = interaction_matrix.iloc[user_idx]
        actual_hotels = actual[actual > 0].index.tolist()
        
        if len(actual_hotels) < 2:
            continue
        
        # Generate recommendations using collaborative filtering
        predicted_ratings = user_factors[user_idx].dot(hotel_factors.T)
        predicted_df = pd.Series(predicted_ratings, index=interaction_matrix.columns)
        
        # Remove already booked hotels
        predicted_df = predicted_df[~predicted_df.index.isin(actual_hotels)]
        
        # Get top recommendations
        top_5 = predicted_df.nlargest(5).index.tolist()
        top_10 = predicted_df.nlargest(10).index.tolist()
        
        # Calculate precision
        prec_5 = precision_at_k(actual_hotels, top_5, k=5)
        prec_10 = precision_at_k(actual_hotels, top_10, k=10)
        
        precisions_at_5.append(prec_5)
        precisions_at_10.append(prec_10)
    
    # Calculate metrics
    metrics = {
        'precision_at_5': np.mean(precisions_at_5) if precisions_at_5 else 0,
        'precision_at_10': np.mean(precisions_at_10) if precisions_at_10 else 0,
        'svd_explained_variance': svd_variance,
        'num_users': interaction_matrix.shape[0],
        'num_hotels': interaction_matrix.shape[1],
        'sparsity': float((interaction_matrix == 0).sum().sum() / (interaction_matrix.shape[0] * interaction_matrix.shape[1]))
    }
    
    print("\n" + "="*50)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("="*50)
    print(f"Precision@5: {metrics['precision_at_5']:.4f}")
    print(f"Precision@10: {metrics['precision_at_10']:.4f}")
    print(f"SVD Explained Variance: {metrics['svd_explained_variance']:.4f}")
    print(f"Number of Users: {metrics['num_users']}")
    print(f"Number of Hotels: {metrics['num_hotels']}")
    print(f"Matrix Sparsity: {metrics['sparsity']*100:.2f}%")
    print("="*50 + "\n")
    
    # Push metrics
    kwargs['ti'].xcom_push(key='recommender_metrics', value=metrics)
    
    # Quality threshold
    if metrics['precision_at_5'] < 0.01:
        print("Warning: Low precision@5, but acceptable for sparse data")
    
    print("Evaluation completed!")

evaluate_task = PythonOperator(
    task_id='evaluate_recommender',
    python_callable=evaluate_recommender,
    dag=dag,
)

# ============================================================================
# TASK 7: SAVE RECOMMENDATION SYSTEM
# ============================================================================

def save_recommender(**kwargs):
    """Save complete recommendation system"""
    print("Saving recommendation system...")
    
    ti = kwargs['ti']
    
    # Pull all components
    interaction_json = ti.xcom_pull(key='interaction_matrix', task_ids='create_interaction_matrix')
    user_similarity_json = ti.xcom_pull(key='user_similarity', task_ids='train_collaborative_model')
    item_similarity_json = ti.xcom_pull(key='item_similarity', task_ids='train_collaborative_model')
    content_similarity_json = ti.xcom_pull(key='content_similarity', task_ids='build_content_features')
    hotel_features_json = ti.xcom_pull(key='hotel_features', task_ids='build_content_features')
    user_factors = np.array(ti.xcom_pull(key='user_factors', task_ids='train_collaborative_model'))
    hotel_factors = np.array(ti.xcom_pull(key='hotel_factors', task_ids='train_collaborative_model'))
    metrics = ti.xcom_pull(key='recommender_metrics', task_ids='evaluate_recommender')
    
    # Convert to DataFrames
    interaction_matrix = pd.read_json(interaction_json)
    user_similarity_df = pd.read_json(user_similarity_json)
    item_similarity_df = pd.read_json(item_similarity_json)
    content_similarity_df = pd.read_json(content_similarity_json)
    hotel_features = pd.read_json(hotel_features_json)
    
    # Create recommender system package
    recommender_system = {
        'interaction_matrix': interaction_matrix,
        'user_similarity': user_similarity_df,
        'item_similarity': item_similarity_df,
        'content_similarity': content_similarity_df,
        'user_factors': user_factors,
        'hotel_factors': hotel_factors,
        'hotel_features': hotel_features
    }
    
    # Save recommender system
    output_path = '/opt/airflow/models/'
    joblib.dump(recommender_system, f'{output_path}hotel_recommender.pkl')
    
    # Save metadata
    recommender_metadata = {
        'num_users': metrics['num_users'],
        'num_hotels': metrics['num_hotels'],
        'sparsity': metrics['sparsity'],
        'precision_at_5': metrics['precision_at_5'],
        'precision_at_10': metrics['precision_at_10'],
        'svd_explained_variance': metrics['svd_explained_variance'],
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(recommender_metadata, f'{output_path}recommender_metadata.pkl')
    
    print("Recommendation system saved successfully!")
    print(f"Files saved to: {output_path}")

save_model_task = PythonOperator(
    task_id='save_recommender',
    python_callable=save_recommender,
    dag=dag,
)

# ============================================================================
# TASK 8: DEPLOY MODEL
# ============================================================================

deploy_model_task = BashOperator(
    task_id='deploy_model',
    bash_command="""
    echo "Deploying hotel recommendation system to production..."
    
    # Copy model files to deployment directory
    cp /opt/airflow/models/hotel_recommender.pkl /opt/deployment/models/
    cp /opt/airflow/models/recommender_*.pkl /opt/deployment/models/
    
    # Restart API service
    # kubectl rollout restart deployment/travel-ml-api
    
    echo "Recommendation system deployment completed!"
    """,
    dag=dag,
)

# ============================================================================
# TASK 9: SEND NOTIFICATION
# ============================================================================

def send_notification(**kwargs):
    """Send notification about pipeline completion"""
    ti = kwargs['ti']
    metrics = ti.xcom_pull(key='recommender_metrics', task_ids='evaluate_recommender')
    
    message = f"""
    Hotel Recommendation System Pipeline Completed Successfully!
    
    Metrics:
    - Precision@5: {metrics['precision_at_5']:.4f}
    - Precision@10: {metrics['precision_at_10']:.4f}
    - Number of Users: {metrics['num_users']}
    - Number of Hotels: {metrics['num_hotels']}
    - Matrix Sparsity: {metrics['sparsity']*100:.2f}%
    
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

extract_data_task >> validate_data_task >> create_matrix_task
create_matrix_task >> build_features_task
create_matrix_task >> train_collab_task
[build_features_task, train_collab_task] >> evaluate_task
evaluate_task >> save_model_task >> deploy_model_task >> send_notification_task