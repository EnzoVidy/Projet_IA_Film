"""
Configuration centralisée pour l'application Filmind.
"""
import os
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "temp_uploads"

# Fichiers
CSV_MOVIES = PROJECT_ROOT / "tmdb_5000_movies.csv"
CSV_CREDITS = PROJECT_ROOT / "tmdb_5000_credits.csv"
MODEL_BOX_OFFICE = PROJECT_ROOT / "box_office_model.pkl"
FINETUNE_OUTPUT = PROJECT_ROOT / "mistral_finetune.jsonl"

# LLM
MISTRAL_MODEL = "mistral-large-latest"
LLM_TEMPERATURE = 0.7

# ML
ML_RANDOM_STATE = 42
ML_N_ESTIMATORS = 100

# Créer les répertoires s'ils n'existent pas
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)