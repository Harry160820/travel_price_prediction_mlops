"""
Flask REST API for Travel MLOps Capstone Project
Serves predictions for Flight Price, Gender Classification, and Hotel Recommendations
Handles missing models gracefully
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import os

# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'notebooks/'
DEBUG_MODE = True

# ============================================================================
# LOAD MODELS AND PREPROCESSORS
# ============================================================================

print("="*80)
print("LOADING MODELS AND PREPROCESSORS")
print("="*80)

models_loaded = {
    'flight_price': False,
    'gender_classification': False,
    'hotel_recommendation': False
}

# ============================================================================
# FLIGHT PRICE PREDICTION MODEL
# ============================================================================
try:
    print("\n[1/3] Loading Flight Price Prediction Model...")
    
    flight_model = joblib.load(f'{MODEL_PATH}flight_price_model.pkl')
    flight_scaler = joblib.load(f'{MODEL_PATH}scaler.pkl')
    flight_features = joblib.load(f'{MODEL_PATH}feature_columns.pkl')
    flight_label_encoders = joblib.load(f'{MODEL_PATH}label_encoders.pkl')
    flight_metadata = joblib.load(f'{MODEL_PATH}model_metadata.pkl')
    
    models_loaded['flight_price'] = True
    print("✓ Flight Price Model loaded successfully")
    print(f"  - Model Type: {flight_metadata.get('model_type', 'Unknown')}")
    print(f"  - Test R²: {flight_metadata.get('metrics', {}).get('test_r2', 'N/A')}")
    
except Exception as e:
    print(f"✗ Error loading Flight Price Model: {str(e)}")
    print("  Make sure these files exist in notebooks/:")
    print("  - flight_price_model.pkl")
    print("  - scaler.pkl")
    print("  - feature_columns.pkl")
    print("  - label_encoders.pkl")
    print("  - model_metadata.pkl")
    flight_model = None

# ============================================================================
# GENDER CLASSIFICATION MODEL (OPTIONAL)
# ============================================================================
try:
    print("\n[2/3] Loading Gender Classification Model...")
    
    gender_model = joblib.load(f'{MODEL_PATH}gender_classifier.pkl')
    gender_scaler = joblib.load(f'{MODEL_PATH}gender_scaler.pkl')
    gender_features = joblib.load(f'{MODEL_PATH}gender_feature_columns.pkl')
    gender_label_encoder = joblib.load(f'{MODEL_PATH}gender_label_encoder.pkl')
    gender_metadata = joblib.load(f'{MODEL_PATH}gender_model_metadata.pkl')
    
    models_loaded['gender_classification'] = True
    print("✓ Gender Classification Model loaded successfully")
    print(f"  - Model Type: {gender_metadata.get('model_type', 'Unknown')}")
    print(f"  - Test Accuracy: {gender_metadata.get('test_accuracy', 'N/A')}")
    
except Exception as e:
    print(f"ℹ Gender Classification Model not available: {str(e)}")
    print("  To enable: Run notebooks/02_classification_model.ipynb")
    models_loaded['gender_classification'] = False
    gender_model = None

# ============================================================================
# HOTEL RECOMMENDATION SYSTEM (OPTIONAL)
# ============================================================================
try:
    print("\n[3/3] Loading Hotel Recommendation System...")
    
    recommender_system = joblib.load(f'{MODEL_PATH}hotel_recommender.pkl')
    recommender_metadata = joblib.load(f'{MODEL_PATH}recommender_metadata.pkl')
    
    models_loaded['hotel_recommendation'] = True
    print("✓ Hotel Recommender loaded successfully")
    print(f"  - Number of Users: {recommender_metadata.get('num_users', 'N/A')}")
    print(f"  - Number of Hotels: {recommender_metadata.get('num_hotels', 'N/A')}")
    
except Exception as e:
    print(f"ℹ Hotel Recommender not available: {str(e)}")
    print("  To enable: Run notebooks/03_recommendation_model.ipynb")
    models_loaded['hotel_recommendation'] = False
    recommender_system = None

print("\n" + "="*80)
print("MODEL LOADING COMPLETE")
print("="*80)
for model_name, loaded in models_loaded.items():
    status = "✓ Loaded" if loaded else "✗ Not Loaded"
    print(f"{model_name}: {status}")
print("="*80 + "\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_encode(value, encoder, default=0):
    """Safely encode a value using LabelEncoder"""
    try:
        return encoder.transform([value])[0]
    except:
        return default

def preprocess_flight_input(data):
    """Preprocess input data for flight price prediction"""
    try:
        features = {}
        
        # Encode categorical variables
        features['from_encoded'] = safe_encode(
            data.get('from', 'Unknown'), 
            flight_label_encoders.get('from')
        )
        features['to_encoded'] = safe_encode(
            data.get('to', 'Unknown'),
            flight_label_encoders.get('to')
        )
        features['flightType_encoded'] = safe_encode(
            data.get('flightType', 'Economy'),
            flight_label_encoders.get('flightType')
        )
        features['agency_encoded'] = safe_encode(
            data.get('agency', 'Unknown'),
            flight_label_encoders.get('agency')
        )
        features['gender_encoded'] = safe_encode(
            data.get('gender', 'Unknown'),
            flight_label_encoders.get('gender')
        )
        features['company_encoded'] = safe_encode(
            data.get('company', 'Unknown'),
            flight_label_encoders.get('company')
        )
        
        # Numeric features
        features['distance'] = float(data.get('distance', 0))
        features['time'] = float(data.get('time', 0))
        features['age'] = int(data.get('age', 30))
        
        # Date features
        if 'date' in data:
            try:
                date_obj = pd.to_datetime(data['date'])
                features['day_of_week'] = date_obj.dayofweek
                features['month'] = date_obj.month
                features['is_weekend'] = 1 if date_obj.dayofweek in [5, 6] else 0
            except:
                features['day_of_week'] = 0
                features['month'] = 1
                features['is_weekend'] = 0
        else:
            features['day_of_week'] = 0
            features['month'] = 1
            features['is_weekend'] = 0
        
        # Derived features (NO price-based features!)
        features['speed'] = features['distance'] / (features['time'] + 1)
        features['distance_time_ratio'] = features['distance'] / (features['time'] + 0.1)
        features['is_long_distance'] = 1 if features['distance'] > 2000 else 0
        features['is_long_duration'] = 1 if features['time'] > 3 else 0
        features['age_group_encoded'] = min(int(features['age'] / 15), 3)
        
        # Create feature array in correct order
        feature_array = []
        for col in flight_features:
            feature_array.append(features.get(col, 0))
        
        return np.array(feature_array).reshape(1, -1)
        
    except Exception as e:
        raise ValueError(f"Error preprocessing flight input: {str(e)}")

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Travel ML API',
        'version': '1.0.0',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': {
                'url': '/health',
                'method': 'GET',
                'description': 'Check API health status'
            },
            'flight_price': {
                'url': '/predict/flight-price',
                'method': 'POST',
                'description': 'Predict flight price',
                'available': models_loaded['flight_price']
            },
            'gender': {
                'url': '/predict/gender',
                'method': 'POST',
                'description': 'Classify user gender',
                'available': models_loaded['gender_classification']
            },
            'hotel_recommendations': {
                'url': '/recommend/hotels',
                'method': 'POST',
                'description': 'Get hotel recommendations',
                'available': models_loaded['hotel_recommendation']
            },
            'model_info': {
                'url': '/model-info',
                'method': 'GET',
                'description': 'Get model information'
            }
        },
        'models_status': models_loaded
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'active',
        'models': models_loaded
    }), 200

@app.route('/predict/flight-price', methods=['POST'])
def predict_flight_price():
    """Predict flight price"""
    try:
        # Check if model is loaded
        if not models_loaded['flight_price']:
            return jsonify({
                'status': 'error',
                'message': 'Flight price prediction model not loaded. Please check server logs.'
            }), 503
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['from', 'to', 'distance', 'time']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Preprocess input
        features = preprocess_flight_input(data)
        
        # Scale features
        features_scaled = flight_scaler.transform(features)
        
        # Make prediction
        prediction = flight_model.predict(features_scaled)[0]
        
        # Ensure prediction is positive
        prediction = max(0, prediction)
        
        # Calculate confidence
        confidence = "high" if 100 < prediction < 2000 else "medium"
        
        # Return response
        response = {
            'status': 'success',
            'predicted_price': round(float(prediction), 2),
            'currency': 'USD',
            'model_confidence': confidence,
            'input_data': {
                'route': f"{data.get('from')} → {data.get('to')}",
                'distance': data.get('distance'),
                'duration': data.get('time'),
                'flight_type': data.get('flightType', 'Economy')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if DEBUG_MODE else None
        }), 500

@app.route('/predict/gender', methods=['POST'])
def predict_gender():
    """Predict user gender based on travel patterns"""
    try:
        # Check if model is loaded
        if not models_loaded['gender_classification']:
            return jsonify({
                'status': 'error',
                'message': 'Gender classification model not loaded. Please train the model by running notebooks/02_classification_model.ipynb'
            }), 503
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400
        
        # Create feature array
        features = {}
        for col in gender_features:
            features[col] = data.get(col, 0)
        
        feature_array = np.array([features[col] for col in gender_features]).reshape(1, -1)
        
        # Scale features
        features_scaled = gender_scaler.transform(feature_array)
        
        # Make prediction
        prediction = gender_model.predict(features_scaled)[0]
        prediction_proba = gender_model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        predicted_gender = gender_label_encoder.inverse_transform([prediction])[0]
        
        # Return response
        response = {
            'status': 'success',
            'predicted_gender': predicted_gender,
            'confidence': {
                gender_label_encoder.classes_[i]: round(float(prob), 4)
                for i, prob in enumerate(prediction_proba)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if DEBUG_MODE else None
        }), 500

@app.route('/recommend/hotels', methods=['POST'])
def recommend_hotels():
    """Get hotel recommendations for a user"""
    try:
        # Check if model is loaded
        if not models_loaded['hotel_recommendation']:
            return jsonify({
                'status': 'error',
                'message': 'Hotel recommendation system not loaded. Please train the model by running notebooks/03_recommendation_model.ipynb'
            }), 503
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400
        
        user_code = data.get('user_code')
        n_recommendations = data.get('n_recommendations', 5)
        method = data.get('method', 'hybrid')
        
        if not user_code:
            return jsonify({'error': 'user_code is required'}), 400
        
        # Get components from recommender system
        interaction_matrix = recommender_system['interaction_matrix']
        user_factors = recommender_system['user_factors']
        hotel_factors = recommender_system['hotel_factors']
        content_similarity = recommender_system['content_similarity']
        
        # Check if user exists
        if user_code not in interaction_matrix.index:
            return jsonify({
                'status': 'error',
                'message': f'User {user_code} not found in database'
            }), 404
        
        # Get user's booking history
        user_bookings = interaction_matrix.loc[user_code]
        booked_hotels = user_bookings[user_bookings > 0].index.tolist()
        
        # Generate recommendations
        if method == 'collaborative' or method == 'hybrid':
            user_idx = interaction_matrix.index.get_loc(user_code)
            predicted_ratings = user_factors[user_idx].dot(hotel_factors.T)
            collab_scores = pd.Series(predicted_ratings, index=interaction_matrix.columns)
            collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min())
        
        if method == 'content' or method == 'hybrid':
            if len(booked_hotels) > 0:
                content_scores = content_similarity[booked_hotels].mean(axis=1)
            else:
                content_scores = pd.Series(0, index=interaction_matrix.columns)
        
        # Combine scores
        alpha = 0.7
        if method == 'hybrid':
            final_scores = alpha * collab_scores + (1 - alpha) * content_scores
        elif method == 'collaborative':
            final_scores = collab_scores
        else:
            final_scores = content_scores
        
        # Remove already booked hotels
        final_scores = final_scores[~final_scores.index.isin(booked_hotels)]
        
        # Get top recommendations
        recommendations = final_scores.nlargest(n_recommendations)
        
        # Format response
        response = {
            'status': 'success',
            'user_code': user_code,
            'method': method,
            'booked_hotels_count': len(booked_hotels),
            'recommendations': [
                {
                    'hotel_name': hotel,
                    'score': round(float(score), 4),
                    'rank': idx + 1
                }
                for idx, (hotel, score) in enumerate(recommendations.items())
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if DEBUG_MODE else None
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    try:
        result = {
            'status': 'success',
            'models': {}
        }
        
        # Flight price model info
        if models_loaded['flight_price']:
            result['models']['flight_price'] = {
                'type': flight_metadata.get('model_type'),
                'test_r2': flight_metadata.get('metrics', {}).get('test_r2'),
                'test_rmse': flight_metadata.get('metrics', {}).get('test_rmse'),
                'training_date': flight_metadata.get('training_date'),
                'available': True
            }
        else:
            result['models']['flight_price'] = {
                'available': False,
                'message': 'Model not loaded'
            }
        
        # Gender classification model info
        if models_loaded['gender_classification']:
            result['models']['gender_classification'] = {
                'type': gender_metadata.get('model_type'),
                'test_accuracy': gender_metadata.get('test_accuracy'),
                'test_f1': gender_metadata.get('test_f1'),
                'training_date': gender_metadata.get('training_date'),
                'available': True
            }
        else:
            result['models']['gender_classification'] = {
                'available': False,
                'message': 'Model not loaded. Run notebooks/02_classification_model.ipynb'
            }
        
        # Hotel recommendation system info
        if models_loaded['hotel_recommendation']:
            result['models']['hotel_recommendation'] = {
                'num_users': recommender_metadata.get('num_users'),
                'num_hotels': recommender_metadata.get('num_hotels'),
                'precision_at_5': recommender_metadata.get('precision_at_5'),
                'training_date': recommender_metadata.get('training_date'),
                'available': True
            }
        else:
            result['models']['hotel_recommendation'] = {
                'available': False,
                'message': 'Model not loaded. Run notebooks/03_recommendation_model.ipynb'
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Could not retrieve model information',
            'details': str(e)
        }), 500

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING FLASK API SERVER")
    print("="*80)
    print(f"Available models: {sum(models_loaded.values())}/3")
    print(f"Server will run on: http://0.0.0.0:5000")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)