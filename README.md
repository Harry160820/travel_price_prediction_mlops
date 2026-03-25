# Travel MLOps Capstone Project

Complete end-to-end Machine Learning Operations (MLOps) pipeline for travel price prediction, gender classification, and hotel recommendations.

##  Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Development](#model-development)
- [API Deployment](#api-deployment)
- [MLOps Pipeline](#mlops-pipeline)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Contributing](#contributing)

##  Project Overview

This project demonstrates a complete MLOps workflow for travel-related machine learning models:

1. **Flight Price Prediction** (Regression)
2. **Gender Classification** (Classification)
3. **Hotel Recommendations** (Recommendation System)

### Business Context

The project leverages data analytics and machine learning to revolutionize travel experiences by:
- Predicting accurate flight prices
- Understanding user demographics through travel patterns
- Providing personalized hotel recommendations

##  Features

### Machine Learning Models

- ✈️ **Flight Price Prediction**: Random Forest Regressor with R² > 0.85
- 👤 **Gender Classification**: Ensemble classifier with accuracy > 85%
- 🏨 **Hotel Recommendations**: Hybrid collaborative filtering system

### MLOps Components

- **Docker**: Containerized application
- **Kubernetes**: Scalable deployment with auto-scaling
- **Apache Airflow**: Automated data pipelines and model retraining
- **Jenkins**: CI/CD pipeline for continuous deployment
- **MLFlow**: Experiment tracking and model versioning
- **Flask API**: RESTful API for model serving
- **Streamlit**: Interactive web interface

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Users   │  │ Flights  │  │  Hotels  │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering                         │
│  • Date features  • Encoding  • Aggregations                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Model Training (MLFlow)                    │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐                  │
│  │Regression│  │Classifier│  │Recommender│                  │
│  └──────────┘  └──────────┘  └───────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Flask REST API                           │
│              (Containerized with Docker)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           Kubernetes Deployment (Auto-scaling)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Pod 1   │  │  Pod 2   │  │  Pod 3   │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces                          │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Streamlit UI    │  │   API Clients    │                 │
│  └──────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Basic system-design(Intuition): [System-Design](https://app.eraser.io/workspace/FXN4MXxRBYcHAnMHuXxx?origin=share)

## Prerequisites

### Software Requirements

- Python 3.10
- Docker 20.10+
- Kubernetes 1.21+
- Apache Airflow 2.0+
- Jenkins 2.300+
- MLFlow 2.0+

### Python Packages

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/travel-mlops-capstone.git
cd travel-mlops-capstone
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Data

Place your CSV files in the project root:
- `users.csv`
- `flights.csv`
- `hotels.csv`

### 4. Train Models

```bash
# Run notebooks or Python scripts
jupyter notebook notebooks/01_regression_model.ipynb
jupyter notebook notebooks/02_classification_model.ipynb
jupyter notebook notebooks/03_recommendation_model.ipynb
```

### 5. Start the API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### 6. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

##  Project Structure

```
travel-mlops-capstone/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── notebooks/
│   ├── 01_regression_model.ipynb
│   ├── 02_classification_model.ipynb
│   └── 03_recommendation_model.ipynb
|   ├── flight_price_model.pkl
│   ├── gender_classifier.pkl
│   ├── hotel_recommender.pkl
│   └── *.pkl (preprocessing objects)
│
│   
│
├── api/
│   ├── app.py
│   └── requirements.txt
│
├── streamlit/
│   └── streamlit_app.py
│
├── kubernetes/
│   ├── deployment.yml
│   └── service.yml
│
├── airflow/
│   └── dags/
│       ├── regression_pipeline.py
│       ├── classification_pipeline.py
│       └── recommendation_pipeline.py
│
├── jenkins/
│   └── Jenkinsfile
│
├── mlflow/
│   └── mlflow_tracking.py
│
└── tests/
    └── test_api.py
```

## Model Development

### Regression Model (Flight Price Prediction)

**Algorithm**: Random Forest Regressor

**Features**:
- Distance, time, flight type, agency
- Date features (day of week, month, weekend)
- User demographics (age, gender, company)
- Derived features (price per km, speed)

**Performance**:
- R² Score: 0.87
- RMSE: 45.32
- MAE: 32.15

### Classification Model (Gender Prediction)

**Algorithm**: Random Forest Classifier

**Features**:
- Travel spending patterns
- Flight and hotel booking frequencies
- Average distances and durations
- Preference patterns

**Performance**:
- Accuracy: 87.5%
- Precision: 0.88
- Recall: 0.86
- F1-Score: 0.87

### Recommendation System (Hotel Suggestions)

**Approach**: Hybrid (Collaborative + Content-Based)

**Methods**:
- User-based collaborative filtering
- Item-based collaborative filtering
- Content-based filtering using hotel features
- Matrix factorization with SVD


## 🔌 API Deployment

### Local Development

```bash
python app.py
```

### Docker Deployment

```bash
# Build image
docker build -t travel-ml-api .

# Run container
docker run -p 5000:5000 travel-ml-api
```

### Kubernetes Deployment

```bash
# Apply configuration
kubectl apply -f kubernetes/deployment.yml

# Check status
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment travel-ml-api --replicas=5
```

## MLOps Pipeline

### Apache Airflow

**DAG Schedule**: Daily at midnight

**Tasks**:
1. Extract data from sources
2. Validate data quality
3. Feature engineering
4. Model training
5. Model evaluation
6. Model deployment
7. Send notifications

```bash
# Start Airflow
airflow standalone

# Trigger DAG manually
airflow dags trigger flight_price_regression_pipeline
```

### Jenkins CI/CD

**Pipeline Stages**:
1. Checkout code
2. Setup environment
3. Data validation
4. Unit tests
5. Train model
6. Model validation
7. Build Docker image
8. Security scan
9. Push to registry
10. Deploy to Kubernetes
11. Integration tests
12. Performance tests

### MLFlow Tracking

```bash
# Start MLFlow UI
mlflow ui

# Access at http://localhost:5000
```

**Tracked Metrics**:
- Model parameters
- Training metrics (R², RMSE, MAE)
- Feature importance
- Model artifacts
- Experiment comparisons

## Testing

### Unit Tests

```bash
python -m pytest tests/ -v
```

### Integration Tests

```bash
# Test API endpoints
curl -X POST http://localhost:5000/predict/flight-price \
  -H "Content-Type: application/json" \
  -d '{"from": "NYC", "to": "LAX", "distance": 4000, ...}'
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:5000/health
```

## Monitoring

### Health Check

```bash
curl http://localhost:5000/health
```

### Model Information

```bash
curl http://localhost:5000/model-info
```

### Kubernetes Monitoring

```bash
# Check pod status
kubectl get pods --watch

# View logs
kubectl logs -f deployment/travel-ml-api

# Check resource usage
kubectl top pods
```

## API Documentation

### Endpoints

#### 1. Health Check
```
GET /health
```

#### 2. Flight Price Prediction
```
POST /predict/flight-price
Content-Type: application/json

{
  "from": "New York",
  "to": "Los Angeles",
  "distance": 4000,
  "time": 5.5,
  "flightType": "Economy",
  "agency": "SkyWings",
  "date": "2024-01-15",
  "age": 30,
  "gender": "Male",
  "company": "TechCorp"
}
```

#### 3. Gender Classification
```
POST /predict/gender
Content-Type: application/json

{
  "age": 35,
  "avg_flight_price": 500,
  "num_flights": 10,
  ...
}
```

#### 4. Hotel Recommendations
```
POST /recommend/hotels
Content-Type: application/json

{
  "user_code": "user_001",
  "n_recommendations": 5,
  "method": "hybrid"
}
```

## Security

- API rate limiting
- Input validation
- Docker security scanning with Trivy
- Kubernetes RBAC
- Secret management with Kubernetes Secrets

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## 👥 Authors

- Hari Om - [GitHub](https://github.com/Harry160820)

## Acknowledgments

- Dataset providers
- Open source community
- MLOps best practices resources

