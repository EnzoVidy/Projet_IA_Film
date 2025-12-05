"""
STRUCTURE DES DÉPENDANCES - Filmind
====================================

📊 DIAGRAMME D'ARCHITECTURE

                            app.py
                              |
                ______________|______________
               |               |              |
          llm_utils.py   ml_utils.py   finetune_prep.py
               |               |              |
          config.py       config.py      config.py
           /    |  \       /    |  \      /    |  \
      paths  params .env  paths  params  paths  params

═══════════════════════════════════════════════════════════

🔗 DÉPENDANCES DÉTAILLÉES

1️⃣  app.py (Main Application)
   ├─ Import: llm_utils
   ├─ Import: ml_utils
   ├─ Import: finetune_prep
   ├─ Import: streamlit
   ├─ Import: pandas
   └─ Dépend de: Configuration centralisée

2️⃣  llm_utils.py (LLM Functions)
   ├─ Import: config
   ├─ Import: python-dotenv (pour .env)
   ├─ Import: langchain
   ├─ Import: langchain-core
   ├─ Import: langchain-mistralai
   └─ Dépend de: .env (MISTRAL_API_KEY)

3️⃣  ml_utils.py (Machine Learning)
   ├─ Import: config
   ├─ Import: pandas
   ├─ Import: scikit-learn
   ├─ Import: joblib
   └─ Dépend de: tmdb_5000_movies.csv

4️⃣  finetune_prep.py (Data Preparation)
   ├─ Import: config
   ├─ Import: pandas
   ├─ Import: json (stdlib)
   └─ Dépend de: tmdb_5000_movies.csv

5️⃣  config.py (Configuration - NEW)
   ├─ Centralise tous les chemins
   ├─ Centralise tous les paramètres
   └─ Crée les répertoires nécessaires

═══════════════════════════════════════════════════════════

📦 STRUCTURE DE FICHIERS

Projet_IA_Film/
├── app.py                      ← Application principale
├── llm_utils.py               ← Fonctions LLM
├── ml_utils.py                ← Modèles Machine Learning
├── finetune_prep.py           ← Préparation fine-tuning
├── config.py                  ← Configuration centralisée (NEW)
├── check_dependencies.py      ← Vérification des dépendances (NEW)
├── __init__.py               ← Package initialization (NEW)
├── README.md                 ← Documentation (NEW)
├── requirements.txt          ← Dépendances Python
├── .env                      ← Configuration locale (API Keys)
├── .gitignore               ← Git configuration
├── __pycache__/             ← Cache Python
├── venv/                    ← Virtual environment
├── tmdb_5000_movies.csv     ← Données source
├── tmdb_5000_credits.csv    ← Données supplémentaires
├── box_office_model.pkl     ← Modèle ML (généré)
└── mistral_finetune.jsonl   ← Fine-tuning data (généré)

═══════════════════════════════════════════════════════════

🔍 VÉRIFICATION

✅ Tous les fichiers Python importent correctement
✅ Toutes les dépendances sont dans requirements.txt
✅ Configuration centralisée dans config.py
✅ Pas de chemins en dur dans le code
✅ Gestion d'erreurs pour .env
✅ Structure modulaire et maintenable

═══════════════════════════════════════════════════════════

🚀 UTILISATION

# Installation
pip install -r requirements.txt

# Vérifier les dépendances
python check_dependencies.py

# Lancer l'app
streamlit run app.py

═══════════════════════════════════════════════════════════
"""

# Imprimer le diagramme
if __name__ == "__main__":
    print(__doc__)
