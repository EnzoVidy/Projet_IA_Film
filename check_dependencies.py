#!/usr/bin/env python3
"""
Script de vérification des dépendances - Filmind
Teste que tous les modules et imports fonctionnent correctement.
"""

import sys
from pathlib import Path

def check_imports():
    """Vérifie que tous les imports nécessaires sont disponibles."""
    
    print("🔍 Vérification des dépendances Filmind...\n")
    
    errors = []
    
    # Test 1: Config
    try:
        import config
        print("✅ config.py - Configuration OK")
    except Exception as e:
        errors.append(f"❌ config.py - {e}")
        print(f"❌ config.py - {e}")
    
    # Test 2: LLM Utils
    try:
        import llm_utils
        print("✅ llm_utils.py - LLM Functions OK")
    except Exception as e:
        errors.append(f"❌ llm_utils.py - {e}")
        print(f"❌ llm_utils.py - {e}")
    
    # Test 3: ML Utils
    try:
        import ml_utils
        print("✅ ml_utils.py - ML Functions OK")
    except Exception as e:
        errors.append(f"❌ ml_utils.py - {e}")
        print(f"❌ ml_utils.py - {e}")
    
    # Test 4: Finetune Prep
    try:
        import finetune_prep
        print("✅ finetune_prep.py - Finetune Prep OK")
    except Exception as e:
        errors.append(f"❌ finetune_prep.py - {e}")
        print(f"❌ finetune_prep.py - {e}")
    
    # Test 5: Streamlit (optionnel, pour développement)
    try:
        import streamlit
        print("✅ streamlit - UI Framework OK")
    except Exception as e:
        print(f"⚠️  streamlit - {e}")
    
    # Test 6: LangChain
    try:
        from langchain_mistralai.chat_models import ChatMistralAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        print("✅ langchain* - LLM Framework OK")
    except Exception as e:
        errors.append(f"❌ langchain* - {e}")
        print(f"❌ langchain* - {e}")
    
    # Test 7: ML Libraries
    try:
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        import pandas
        print("✅ sklearn/joblib/pandas - ML Libraries OK")
    except Exception as e:
        errors.append(f"❌ sklearn/joblib/pandas - {e}")
        print(f"❌ sklearn/joblib/pandas - {e}")
    
    # Test 8: Environment
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key:
            print("✅ .env - MISTRAL_API_KEY trouvée")
        else:
            print("⚠️  .env - MISTRAL_API_KEY manquante (nécessaire pour llm_utils)")
    except Exception as e:
        print(f"⚠️  .env - {e}")
    
    print("\n" + "="*50)
    if errors:
        print(f"❌ {len(errors)} erreur(s) détectée(s):\n")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("✅ Toutes les dépendances sont correctement configurées!")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
