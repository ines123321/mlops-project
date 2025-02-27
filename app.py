from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from model_pipeline import load_data, preprocess_data
from fastapi.middleware.cors import CORSMiddleware
# Charger le modèle
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)
# Debug: Print the type of the loaded model
print(f"Loaded model type: {type(model)}")

# Définir l'application FastAPI
app = FastAPI()




# Définir un modèle de données pour la requête de prédiction
class PredictionInput(BaseModel):

    features: list

# Définir un modèle de données pour la requête de réentraînement
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
        # Charger et prétraiter les données
        df = load_data()
        df_scaled = preprocess_data(df)  # Only one value is returned
        
        # Préparer les données pour l'entraînement
        X = df_scaled
        Y = df['Churn']
        X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Créer le modèle de Random Forest avec les paramètres de la requête
        rf = RandomForestClassifier(
            n_estimators=data.n_estimators,
            max_depth=data.max_depth,
            min_samples_split=data.min_samples_split,
            min_samples_leaf=data.min_samples_leaf,
            bootstrap=data.bootstrap,
            random_state=42
        )

        # Entraîner le modèle
        rf.fit(X_train_scaled, Y_train)

        # Sauvegarder le modèle réentraîné
        joblib.dump(rf, MODEL_PATH)

        return {"message": "Model retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

