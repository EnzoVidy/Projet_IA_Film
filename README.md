# 🎬 Filmind - Suite IA Complète pour Films

## Structure des Dépendances

### Fichiers Principaux

#### 1. **`app.py`** (Interface Streamlit)
   - **Imports depuis `llm_utils`** : Fonctions d'analyse/génération avec Mistral AI
     - `recommander_films()` - Recommandations personnalisées
     - `genre_depuis_synopsis()` - Classification de genre
     - `generer_critique()` - Génération de critiques
     - `generer_synopsis()` - Génération de synopses
     - `generer_casting()` - Proposition de castings
     - `corriger_script()` - Correction de textes
     - `generer_bande_annonce()` - Génération de scripts de bande-annonce
   
   - **Imports depuis `ml_utils`** : Modèles prédictifs
     - `predict_box_office()` - Prédiction des revenus au box-office
     - `train_model()` - Entraînement du modèle Random Forest
   
   - **Imports depuis `finetune_prep`** : Préparation fine-tuning
     - `prepare_mistral_finetuning_data()` - Génération du fichier JSONL

#### 2. **`llm_utils.py`** (LLM Functions - Mistral AI)
   - Importe : `config.py` (MISTRAL_MODEL, LLM_TEMPERATURE)
   - Importe : `.env` (MISTRAL_API_KEY)
   - Dépend de : `langchain`, `langchain-core`, `langchain-mistralai`, `python-dotenv`

#### 3. **`ml_utils.py`** (Machine Learning)
   - Importe : `config.py` (MODEL_BOX_OFFICE, CSV_MOVIES, ML_RANDOM_STATE, ML_N_ESTIMATORS)
   - Charge le CSV : `tmdb_5000_movies.csv`
   - Sauvegarde : `box_office_model.pkl`
   - Dépend de : `pandas`, `scikit-learn`, `joblib`

#### 4. **`finetune_prep.py`** (Data Preparation)
   - Importe : `config.py` (CSV_MOVIES, FINETUNE_OUTPUT)
   - Charge le CSV : `tmdb_5000_movies.csv`
   - Génère : `mistral_finetune.jsonl`
   - Dépend de : `pandas`, `json` (stdlib)

#### 5. **`config.py`** (Configuration Centralisée - NOUVEAU)
   - Point central de configuration
   - Définit les chemins des fichiers
   - Centralise les paramètres de modèles

#### 6. **`__init__.py`** (Package initialization - NOUVEAU)
   - Documente le package
   - Facilite les imports de sous-modules

### Fichiers de Données

- **`tmdb_5000_movies.csv`** : Source pour entraînement ML et fine-tuning
- **`tmdb_5000_credits.csv`** : Données supplémentaires (actuellement non utilisées)
- **`.env`** : Configuration locale (MISTRAL_API_KEY)

### Fichiers Générés

- **`box_office_model.pkl`** : Modèle ML entraîné (généré par `ml_utils.py`)
- **`mistral_finetune.jsonl`** : Données fine-tuning (généré par `finetune_prep.py`)

### Dépendances Python

```
streamlit           # Interface Web
langchain           # Orchestration LLM
langchain-core      # Core LLM abstractions
langchain-mistralai # Mistral AI provider
python-dotenv       # Gestion .env
pandas              # Data manipulation (ML + fine-tuning prep)
scikit-learn        # Machine Learning algorithms
joblib              # Model serialization
```

### Graphique de Dépendances

```
app.py
├── llm_utils.py
│   ├── config.py
│   ├── .env
│   └── [langchain dependencies]
├── ml_utils.py
│   ├── config.py
│   ├── tmdb_5000_movies.csv
│   └── [pandas, scikit-learn, joblib]
└── finetune_prep.py
    ├── config.py
    ├── tmdb_5000_movies.csv
    └── [pandas, json]

config.py
├── Centralise paths et parametres
└── Crée les repertoires necessaires
```

## Installation & Lancement

### 1. Installation des dépendances
```bash
cd /home/flood/Documents/iut/s5/iacine/Projet_IA_Film
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Assurez-vous que `.env` contient :
```
MISTRAL_API_KEY=votre_clé_api
```

### 3. Lancement
```bash
streamlit run app.py
```

Accédez à : `http://localhost:8501`

## Vérification des Imports

✅ Tous les imports sont correctement configurés
✅ Toutes les dépendances sont listées dans `requirements.txt`
✅ Les chemins sont centralisés dans `config.py`
✅ Le fichier `.env` est en place

