import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from config import MODEL_BOX_OFFICE, CSV_MOVIES, ML_RANDOM_STATE

# Chemins configurés
MODEL_PATH = str(MODEL_BOX_OFFICE)
DATA_PATH = str(CSV_MOVIES)

def train_and_compare_models():
    """
    Entraîne plusieurs modèles, les compare et sauvegarde le meilleur.
    Retourne un DataFrame de résultats et le nom du meilleur modèle.
    """
    if not os.path.exists(DATA_PATH):
        return "Erreur : CSV introuvable.", None

    # 1. Chargement et nettoyage
    df = pd.read_csv(DATA_PATH)
    df = df[['budget', 'runtime', 'revenue']].dropna()
    # Filtrage des données aberrantes (budget ou revenu nul)
    df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)]

    X = df[['budget', 'runtime']]
    y = df['revenue']

    # 2. Split Train/Test (Essentiel pour la comparaison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=ML_RANDOM_STATE
    )

    # 3. Définition des modèles à comparer
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=ML_RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=ML_RANDOM_STATE)
    }

    results = []
    best_score = -float("inf")
    best_model = None
    best_model_name = ""

    # 4. Entraînement et Évaluation
    for name, model in models.items():
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction sur le set de test
        predictions = model.predict(X_test)
        
        # Calcul des métriques
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        results.append({
            "Modèle": name,
            "R2 Score": round(r2, 4), # Plus proche de 1 est mieux
            "MAE ($)": round(mae, 2)  # Plus bas est mieux
        })

        # On garde le modèle avec le meilleur R2
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    # 5. Sauvegarde du champion
    if best_model:
        joblib.dump(best_model, MODEL_PATH)

    return pd.DataFrame(results), best_model_name

def predict_box_office(budget, runtime):
    """Charge le modèle (le meilleur sauvegardé) et fait une prédiction."""
    if not os.path.exists(MODEL_PATH):
        return None 
    
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([[budget, runtime]])
    return prediction[0]