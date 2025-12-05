import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from config import MODEL_BOX_OFFICE, CSV_MOVIES, ML_RANDOM_STATE, ML_N_ESTIMATORS

# Chemins configurés
MODEL_PATH = str(MODEL_BOX_OFFICE)
DATA_PATH = str(CSV_MOVIES)

def train_model():
    """Charge les données, entraine le modèle et le sauvegarde."""
    if not os.path.exists(DATA_PATH):
        return "Erreur : Fichier csv introuvable. Téléchargez tmdb_5000_movies.csv"

    # 1. Chargement et nettoyage
    df = pd.read_csv(DATA_PATH)
    
    # On garde uniquement les colonnes utiles et on retire les lignes avec des valeurs nulles ou zéro
    df = df[['budget', 'runtime', 'revenue']].dropna()
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

    X = df[['budget', 'runtime']]
    y = df['revenue']

    # 2. Entraînement (Random Forest est robuste pour ça)
    model = RandomForestRegressor(n_estimators=ML_N_ESTIMATORS, random_state=ML_RANDOM_STATE)
    model.fit(X, y)

    # 3. Sauvegarde
    joblib.dump(model, MODEL_PATH)
    return "Modèle entraîné et sauvegardé avec succès !"

def predict_box_office(budget, runtime):
    """Charge le modèle et fait une prédiction."""
    if not os.path.exists(MODEL_PATH):
        # Si le modèle n'existe pas, on tente de l'entraîner à la volée
        res = train_model()
        if "Erreur" in res:
            return None
    
    model = joblib.load(MODEL_PATH)
    
    # Prédiction
    prediction = model.predict([[budget, runtime]])
    return prediction[0]