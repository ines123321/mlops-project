
# MLOps Churn Prediction Pipeline

## Project Overview
This project implements an end-to-end **MLOps pipeline for customer churn prediction**.  
It covers data preprocessing, model training, experiment tracking, API deployment, and model retraining using MLOps tools and best practices.

The goal is to predict whether a customer is likely to churn based on historical behavioral data.

---

## Features
- **Churn Prediction Model** using Random Forest
- **Automated Feature Engineering** and data preprocessing
- **Model Training & Evaluation** with GridSearchCV
- **Experiment Tracking** using MLflow
- **REST API Deployment** with FastAPI
- **Model Retraining Endpoint**
- **Model Persistence** with joblib
- **Containerization** with Docker
- **Pipeline Automation** via CLI arguments

---


---

## Machine Learning Pipeline
- Data loading and cleaning
- Feature scaling and preprocessing
- Optional PCA and clustering (KMeans, Hierarchical)
- Random Forest training with hyperparameter tuning
- Model evaluation (accuracy, ROC, confusion matrix)
- Model saving and versioning

---

## API Endpoints

### Predict Churn
```http
POST /predict/
````

Request body:

```json
{
  "features": [value1, value2, value3, ...]
}
```

### Retrain Model

```http
POST /retrain/
```

Allows retraining the model with custom hyperparameters.

---

## Experiment Tracking

* MLflow is used to:

  * Track parameters
  * Log models
  * Manage experiments
* Each training run is recorded under the **Churn Prediction** experiment.

---

## Tech Stack

* **Python**
* **Scikit-learn**
* **FastAPI**
* **MLflow**
* **Docker**
* **Pandas / NumPy**
* **Matplotlib / Seaborn**

---

## How to Run

### Local

```bash
pip install -r requirements.txt
python app.py
```

### Docker

```bash
docker build -t churn-mlops .
docker run -p 8000:8000 churn-mlops
```


Just tell me 🚀
```
