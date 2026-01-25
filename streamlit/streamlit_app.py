"""
Streamlit Web Application for Travel MLOps Capstone Project
Interactive UI for Flight Price Prediction, Gender Classification, and Hotel Recommendations
Handles missing models gracefully
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Travel ML Predictions",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')
st.sidebar.info(f"🔗 API: {API_BASE_URL}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.sidebar.error(f"API Error: {str(e)}")
        return False

def check_model_availability():
    """Check which models are available"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', {})
            return {
                'flight_price': models.get('flight_price', {}).get('available', False),
                'gender': models.get('gender_classification', {}).get('available', False),
                'hotels': models.get('hotel_recommendation', {}).get('available', False)
            }
    except:
        pass
    return {'flight_price': False, 'gender': False, 'hotels': False}

def predict_flight_price(data):
    """Call flight price prediction API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/flight-price",
            json=data,
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {'status': 'error', 'message': f'Connection error: {str(e)}'}

def predict_gender(data):
    """Call gender classification API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/gender",
            json=data,
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {'status': 'error', 'message': f'Connection error: {str(e)}'}

def get_hotel_recommendations(data):
    """Call hotel recommendation API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/hotels",
            json=data,
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {'status': 'error', 'message': f'Connection error: {str(e)}'}

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">✈️ Travel ML Predictions Platform</div>', 
                unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    # Status indicator in sidebar
    if api_status:
        st.sidebar.success("✅ API Status: Online")
        models_available = check_model_availability()
        
        st.sidebar.markdown("### 📊 Models Status")
        st.sidebar.write(f"✈️ Flight Price: {'✅' if models_available['flight_price'] else '❌'}")
        st.sidebar.write(f"👤 Gender Class: {'✅' if models_available['gender'] else '❌'}")
        st.sidebar.write(f"🏨 Hotels Rec: {'✅' if models_available['hotels'] else '❌'}")
    else:
        st.sidebar.error("❌ API Status: Offline")
        st.error("⚠️ Cannot connect to Flask API. Please ensure:")
        st.code("""
# If running locally:
python api/app.py

# If using Docker:
docker compose up
        """)
        st.info(f"Trying to connect to: {API_BASE_URL}")
        return
    
    # Sidebar Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Service",
        ["🏠 Home", "✈️ Flight Price Prediction", "👤 Gender Classification", 
         "🏨 Hotel Recommendations", "📊 Model Information"]
    )
    
    # ========================================================================
    # HOME PAGE
    # ========================================================================
    
    if page == "🏠 Home":
        st.markdown('<div class="sub-header">Welcome to Travel ML Platform</div>', 
                    unsafe_allow_html=True)
        
        st.write("""
        This platform provides AI-powered predictions and recommendations for travel:
        
        - **Flight Price Prediction**: Get accurate price estimates for your flights
        - **Gender Classification**: Analyze travel patterns to predict user demographics
        - **Hotel Recommendations**: Personalized hotel suggestions based on preferences
        """)
        
        col1, col2, col3 = st.columns(3)
        
        models_count = sum(models_available.values())
        
        with col1:
            st.metric("Services Available", "3")
        with col2:
            st.metric("Models Loaded", f"{models_count}/3")
        with col3:
            st.metric("API Status", "Online" if api_status else "Offline")
        
        st.markdown("---")
        
        if models_count < 3:
            st.warning("⚠️ Some models are not loaded. To enable all features:")
            st.info("""
            **Missing Models? Run these notebooks:**
            - Gender Classification: `notebooks/02_classification_model.ipynb`
            - Hotel Recommendations: `notebooks/03_recommendation_model.ipynb`
            
            After training, restart Docker: `docker compose restart flask-api`
            """)
        
        st.info("👈 Select a service from the sidebar to get started!")
    
    # ========================================================================
    # FLIGHT PRICE PREDICTION
    # ========================================================================
    
    elif page == "✈️ Flight Price Prediction":
        st.markdown('<div class="sub-header">Flight Price Prediction</div>', 
                    unsafe_allow_html=True)
        
        if not models_available['flight_price']:
            st.error("❌ Flight price prediction model is not loaded!")
            st.info("""
            **To enable this feature:**
            1. Run notebook: `notebooks/01_regression_model.ipynb`
            2. Ensure these files are created in notebooks/:
               - flight_price_model.pkl
               - scaler.pkl
               - feature_columns.pkl
               - label_encoders.pkl
               - model_metadata.pkl
            3. Restart Flask: `docker compose restart flask-api`
            """)
            return
        
        st.write("Get instant price predictions for your flight based on multiple factors.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            from_city = st.text_input("From City", value="New York")
            to_city = st.text_input("To City", value="Los Angeles")
            flight_type = st.selectbox("Flight Type", 
                                      ["Economy", "Business", "First Class", "Premium Economy"])
            agency = st.text_input("Preferred Agency", value="SkyWings")
            
        with col2:
            distance = st.number_input("Distance (km)", min_value=0, value=4000, step=100)
            time = st.number_input("Flight Time (hours)", min_value=0.0, value=5.5, step=0.5)
            travel_date = st.date_input("Travel Date", value=date.today())
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
        
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        company = st.text_input("Company", value="TechCorp")
        
        if st.button("🔮 Predict Price", type="primary"):
            with st.spinner("Calculating price..."):
                data = {
                    "from": from_city,
                    "to": to_city,
                    "flightType": flight_type,
                    "agency": agency,
                    "distance": distance,
                    "time": time,
                    "date": travel_date.strftime("%Y-%m-%d"),
                    "age": age,
                    "gender": gender,
                    "company": company
                }
                
                result = predict_flight_price(data)
                
                if result.get('status') == 'success':
                    st.success("✅ Prediction Successful!")
                    
                    st.markdown(f"### Predicted Price: ${result['predicted_price']:.2f}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Route", f"{from_city} → {to_city}")
                    with col2:
                        st.metric("Distance", f"{distance} km")
                    with col3:
                        st.metric("Duration", f"{time} hours")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['predicted_price'],
                        title={'text': "Flight Price (USD)"},
                        delta={'reference': distance * 0.15},
                        gauge={
                            'axis': {'range': [None, result['predicted_price'] * 1.5]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, result['predicted_price'] * 0.5], 'color': "lightgray"},
                                {'range': [result['predicted_price'] * 0.5, result['predicted_price']], 'color': "gray"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
    
    # ========================================================================
    # GENDER CLASSIFICATION
    # ========================================================================
    
    elif page == "👤 Gender Classification":
        st.markdown('<div class="sub-header">Gender Classification</div>', 
                    unsafe_allow_html=True)
        
        if not models_available['gender']:
            st.error("❌ Gender classification model is not loaded!")
            st.info("""
            **To enable this feature:**
            1. Run notebook: `notebooks/02_classification_model.ipynb`
            2. Ensure these files are created in notebooks/:
               - gender_classifier.pkl
               - gender_scaler.pkl
               - gender_feature_columns.pkl
               - gender_label_encoder.pkl
               - gender_model_metadata.pkl
            3. Restart Flask: `docker compose restart flask-api`
            """)
            st.markdown("---")
            st.info("💡 **Note:** This feature is optional. You can complete the project with just flight price prediction.")
            return
        
        st.write("Predict user gender based on travel behavior patterns.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            avg_flight_price = st.number_input("Average Flight Price ($)", 
                                              min_value=0, value=500, step=50)
            num_flights = st.number_input("Number of Flights", 
                                         min_value=0, value=10, step=1)
            total_flight_spent = st.number_input("Total Flight Spending ($)", 
                                                min_value=0, value=5000, step=100)
        
        with col2:
            avg_distance = st.number_input("Average Distance (km)", 
                                          min_value=0, value=2000, step=100)
            num_hotel_bookings = st.number_input("Hotel Bookings", 
                                                min_value=0, value=5, step=1)
            total_travel_spent = st.number_input("Total Travel Spending ($)", 
                                                min_value=0, value=8000, step=100)
            avg_hotel_days = st.number_input("Average Hotel Stay (days)", 
                                           min_value=0, value=3, step=1)
        
        if st.button("🔍 Classify Gender", type="primary"):
            with st.spinner("Analyzing travel patterns..."):
                data = {
                    "age": age,
                    "avg_flight_price": avg_flight_price,
                    "num_flights": num_flights,
                    "total_flight_spent": total_flight_spent,
                    "avg_distance": avg_distance,
                    "num_hotel_bookings": num_hotel_bookings,
                    "total_travel_spent": total_travel_spent,
                    "avg_hotel_days": avg_hotel_days
                }
                
                result = predict_gender(data)
                
                if result.get('status') == 'success':
                    st.success("✅ Classification Successful!")
                    st.markdown(f"### Predicted Gender: {result['predicted_gender']}")
                    
                    confidence = result['confidence']
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(confidence.keys()),
                            y=list(confidence.values()),
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(confidence)]
                        )
                    ])
                    fig.update_layout(
                        title="Classification Confidence",
                        xaxis_title="Gender",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
    
    # ========================================================================
    # HOTEL RECOMMENDATIONS
    # ========================================================================
    
    elif page == "🏨 Hotel Recommendations":
        st.markdown('<div class="sub-header">Hotel Recommendations</div>', 
                    unsafe_allow_html=True)
        
        if not models_available['hotels']:
            st.error("❌ Hotel recommendation system is not loaded!")
            st.info("""
            **To enable this feature:**
            1. Run notebook: `notebooks/03_recommendation_model.ipynb`
            2. Ensure these files are created in notebooks/:
               - hotel_recommender.pkl
               - recommender_metadata.pkl
            3. Restart Flask: `docker compose restart flask-api`
            """)
            st.markdown("---")
            st.info("💡 **Note:** This feature is optional. You can complete the project with just flight price prediction.")
            return
        
        st.write("Get personalized hotel recommendations based on your preferences.")
        
        user_code = st.number_input("User Code"+"1", min_value=0, max_value=1339, value=100, step=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_recommendations = st.slider("Number of Recommendations", 
                                         min_value=1, max_value=20, value=5)
        
        with col2:
            method = st.selectbox("Recommendation Method", 
                                 ["hybrid", "collaborative", "content"])
        
        st.info("""
        **Methods:**
        - **Hybrid**: Combines collaborative and content-based filtering
        - **Collaborative**: Based on similar users' preferences
        - **Content**: Based on hotel features and your history
        """)
        
        if st.button("🎯 Get Recommendations", type="primary"):
            with st.spinner("Finding perfect hotels for you..."):
                data = {
                    "user_code": user_code,
                    "n_recommendations": n_recommendations,
                    "method": method
                }
                
                result = get_hotel_recommendations(data)
                
                if result.get('status') == 'success':
                    st.success("✅ Recommendations Generated!")
                    
                    st.write(f"**User:** {user_code}")
                    st.write(f"**Method:** {method.capitalize()}")
                    st.write(f"**Previous Bookings:** {result['booked_hotels_count']}")
                    
                    st.markdown("### 🏨 Recommended Hotels")
                    
                    recommendations = result['recommendations']
                    
                    if recommendations:
                        df = pd.DataFrame(recommendations)
                        st.dataframe(
                            df.style.background_gradient(subset=['score'], cmap='YlGnBu'),
                            use_container_width=True
                        )
                        
                        fig = px.bar(
                            df,
                            x='score',
                            y='hotel_name',
                            orientation='h',
                            title='Recommendation Scores',
                            labels={'score': 'Recommendation Score', 'hotel_name': 'Hotel'},
                            color='score',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No recommendations available for this user.")
                else:
                    st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
    
    # ========================================================================
    # MODEL INFORMATION
    # ========================================================================
    
    elif page == "📊 Model Information":
        st.markdown('<div class="sub-header">Model Information</div>', 
                    unsafe_allow_html=True)
        
        try:
            response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
            if response.status_code == 200:
                model_info = response.json()
                
                if model_info.get('status') == 'success':
                    models = model_info['models']
                    
                    # Flight Price Model
                    st.markdown("### ✈️ Flight Price Prediction Model")
                    if models['flight_price'].get('available'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model Type", models['flight_price'].get('type', 'N/A'))
                        with col2:
                            st.metric("R² Score", f"{models['flight_price'].get('test_r2', 0):.4f}")
                        with col3:
                            st.metric("RMSE", f"{models['flight_price'].get('test_rmse', 0):.2f}")
                    else:
                        st.warning("❌ Not loaded - " + models['flight_price'].get('message', ''))
                    
                    st.markdown("---")
                    
                    # Gender Classification Model
                    st.markdown("### 👤 Gender Classification Model")
                    if models.get('gender_classification', {}).get('available'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model Type", models['gender_classification'].get('type', 'N/A'))
                        with col2:
                            st.metric("Accuracy", f"{models['gender_classification'].get('test_accuracy', 0):.4f}")
                        with col3:
                            st.metric("F1 Score", f"{models['gender_classification'].get('test_f1', 0):.4f}")
                    else:
                        st.warning("❌ Not loaded - " + models.get('gender_classification', {}).get('message', ''))
                    
                    st.markdown("---")
                    
                    # Hotel Recommendation System
                    st.markdown("### 🏨 Hotel Recommendation System")
                    if models.get('hotel_recommendation', {}).get('available'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Users", f"{models['hotel_recommendation'].get('num_users', 0):,}")
                        with col2:
                            st.metric("Total Hotels", f"{models['hotel_recommendation'].get('num_hotels', 0):,}")
                        with col3:
                            st.metric("Precision@5", f"{models['hotel_recommendation'].get('precision_at_5', 0):.4f}")
                    else:
                        st.warning("❌ Not loaded - " + models.get('hotel_recommendation', {}).get('message', ''))
                        
        except Exception as e:
            st.error(f"Could not fetch model information: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: gray;'>
            Travel ML Platform | MLOps Capstone Project | 2026<br>
            API: {API_BASE_URL} | Models Loaded: {sum(models_available.values())}/3
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()