import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from config import MODEL_BOX_OFFICE, CSV_MOVIES, ML_RANDOM_STATE

# Chemins
MODEL_PATH = str(MODEL_BOX_OFFICE)
DATA_PATH = str(CSV_MOVIES)
ENCODER_PATH = "models/genre_encoder.pkl"

def train_and_compare_models():
    """Entraîne et compare plusieurs modèles avec des variables enrichies."""
    if not os.path.exists(DATA_PATH):
        return "Erreur : CSV introuvable.", None

    # 1. Chargement et Nettoyage
    df = pd.read_csv(DATA_PATH)
    # On ajoute des colonnes pour monter en précision
    cols = ['budget', 'runtime', 'revenue', 'popularity', 'release_date', 'genres']
    df = df[cols].dropna()
    df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)]

    # 2. FEATURE ENGINEERING (Pour dépasser les 50%)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_month'] = df['release_date'].dt.month
    
    def get_first_genre(genre_str):
        try:
            g = json.loads(genre_str)
            return g[0]['name'] if g else 'Unknown'
        except: return 'Unknown'
    
    df['main_genre'] = df['genres'].apply(get_first_genre)
    le = LabelEncoder()
    df['main_genre_encoded'] = le.fit_transform(df['main_genre'])
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, ENCODER_PATH)

    # 3. Préparation (X et y)
    X = df[['budget', 'runtime', 'popularity', 'release_month', 'main_genre_encoded']]
    # Transformation LOG pour réduire l'impact des blockbusters extrêmes
    y = np.log1p(df['revenue']) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=ML_RANDOM_STATE)

    # 4. COMPARAISON DES MODÈLES
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=ML_RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=ML_RANDOM_STATE)
    }

    results = []
    best_r2 = -float("inf")
    best_model = None
    best_model_name = ""

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        # On calcule la MAE sur les vrais dollars (inversion du log)
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
        
        results.append({
            "Modèle": name,
            "R2 Score": round(r2, 4),
            "Erreur Moyenne ($)": round(mae, 2)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model = mdl
            best_model_name = name

    # Sauvegarde du gagnant
    joblib.dump(best_model, MODEL_PATH)

    return pd.DataFrame(results), best_model_name

def predict_box_office(budget, runtime):
    """Prédit le revenu avec le modèle sauvegardé (le meilleur du comparatif)."""
    if not os.path.exists(MODEL_PATH):
        return None 
    
    model = joblib.load(MODEL_PATH)
    
    # Valeurs par défaut pour les nouvelles colonnes
    default_popularity = 21.4 
    default_month = 12
    default_genre = 0 
    
    # Création du vecteur d'entrée avec les bons noms de colonnes
    X_input = pd.DataFrame(
        [[budget, runtime, default_popularity, default_month, default_genre]], 
        columns=['budget', 'runtime', 'popularity', 'release_month', 'main_genre_encoded']
    )
    
    # Prédiction (inversion du Log : exp(x) - 1)
    pred_log = model.predict(X_input)
    return np.expm1(pred_log[0])