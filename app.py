from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from model_pipeline import train_random_forest, load_data, preprocess_data
from sklearn.model_selection import train_test_split

# Charger le modèle
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)
# Debug: Print the type of the loaded model
print(f"Loaded model type: {type(model)}")
# Définir l'application FastAPI
app = FastAPI()

# Définir un modèle de données pour la requête
class PredictionInput(BaseModel):
    features: list

class RetrainInput(BaseModel):
    n_estimators: int = 50
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = False

@app.post("/predict/")
def predict(data: PredictionInput):
    try:
        # Convertir les données en DataFrame
        input_data = pd.DataFrame([data.features])
        
        # Effectuer la prédiction
        prediction = model.predict(input_data)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain/")
def retrain(data: RetrainInput):
    try:
        # Load and preprocess data
        df = load_data()
        df_scaled, scaler = preprocess_data(df)

        # Prepare data for training
        X = df_scaled
        Y = df['Churn']
        X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Train the model with new parameters
        param_grid_rf = {
            'n_estimators': [data.n_estimators],
            'max_depth': [data.max_depth],
            'min_samples_split': [data.min_samples_split],
            'min_samples_leaf': [data.min_samples_leaf],
            'bootstrap': [data.bootstrap]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=1)
        grid_search_rf.fit(X_train_scaled, Y_train)

        best_rf = grid_search_rf.best_estimator_

        # Save the new model
        joblib.dump(best_rf, MODEL_PATH)

        return {"message": "Model retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


