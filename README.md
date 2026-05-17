**Sales Prediction MLOps System
Project Overview**

This project is an end-to-end MLOps Sales Prediction System developed using Machine Learning, FastAPI, Docker, MLflow, GitHub Actions CI/CD, and AWS EC2 deployment.

The system predicts retail sales amounts based on product and transaction features such as:

Unit Price
Quantity
Brand
Category

This project demonstrates a complete production-style MLOps workflow from model training to cloud deployment and automated CI/CD.

Key Features
End-to-End Machine Learning Pipeline
Polynomial Regression Model
Feature Engineering & Preprocessing Pipeline
MLflow Experiment Tracking
FastAPI Prediction API
Docker Containerization
AWS EC2 Cloud Deployment
GitHub Actions CI/CD Automation
REST API Swagger Documentation
Production-style Model Serving Workflow
Tech Stack
Machine Learning
Python
Scikit-learn
Pandas
NumPy
MLOps & Deployment
MLflow
FastAPI
Docker
GitHub Actions
AWS EC2
CI/CD
GitHub Actions
SSH Deployment Automation

**Machine Learning Workflow**
Data Preprocessing
    ↓
Feature Engineering
    ↓
Polynomial Feature Transformation
    ↓
Model Training
    ↓
Model Evaluation
    ↓
MLflow Experiment Tracking
    ↓
Model Serialization (.pkl)
    ↓
FastAPI Model Serving
    ↓
Docker Containerization
    ↓
AWS EC2 Deployment
    ↓
CI/CD Automation
Model Features

The model uses:

Unit Price
Quantity
Brand
Category

Target Variable:

sales_amount_log

using log transformation:

np.log1p(sales_amount)
Evaluation Metrics

The project tracks multiple machine learning metrics:

R² Score
RMSE
MAE

All metrics are logged and monitored using MLflow.

**MLOps Architecture**
Frontend/UI
     ↓
FastAPI Prediction API
     ↓
Docker Container
     ↓
AWS EC2 Deployment

Training Pipeline
     ↓
MLflow Tracking Server
     ↓
Experiment Monitoring
CI/CD Pipeline

**The project includes automated deployment using GitHub Actions.**

Workflow:

git push
    ↓
GitHub Actions Trigger
    ↓
SSH into AWS EC2
    ↓
Pull Latest Source Code
    ↓
Rebuild Docker Container
    ↓
Restart FastAPI Service
API Endpoint
Prediction Endpoint
POST /predict

Example Input:

{
  "unit_price": 120,
  "quantity": 3,
  "brand": "Nike",
  "category": "Shoes"
}

Example Output:

{
  "predicted_sales": 48.63
}
MLflow Experiment Tracking

MLflow is used for:

Experiment Tracking
Parameter Logging
Metric Logging
Model Monitoring
Run Comparison
Future Improvements
Model Registry
Automated Retraining Pipeline
Drift Detection
Kubernetes Deployment
AWS ECR/ECS Integration
Monitoring Dashboard
Frontend React Application
Real-time Prediction Streaming
Project Goals

This project was built to demonstrate practical skills in:

Machine Learning Engineering
MLOps Workflow
Cloud Deployment
CI/CD Automation
API Development
Containerization
Production-Level ML Systems
Author

Zul
Data Scientist & MLOps Enthusiast
