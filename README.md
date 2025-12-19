Doc générée par IA
# Filmind : Intelligence Artificielle appliquée au cycle de vie du film

**Filmind** est une station de travail unifiée permettant d'exploiter les modèles de langage (LLM) et l'apprentissage automatique (ML) pour l'analyse de données cinématographiques, l'aide à l'écriture et la planification de production.

---

## 🛠 Architecture du Projet

Le système repose sur une architecture hybride combinant traitement de données structurées et analyse de texte non structuré.

### Composantes LLM (Mistral AI & LangChain)

* **Agent de routage d'intention :** Utilise des sorties structurées pour diriger les requêtes utilisateurs vers les outils spécifiques (critique, recommandation, analyse).
* **Moteur RAG (Retrieval-Augmented Generation) :** Indexation de scénarios PDF via FAISS et embeddings HuggingFace pour permettre une interrogation contextuelle du script.
* **Analyse de Marché :** Intégration de l'API Tavily pour effectuer des recherches de concurrence en temps réel sur le web.
* **Fine-tuning :** Pipeline de préparation de données (JSONL) pour spécialiser des modèles Mistral sur la génération de loglines et de synopsis.

### Composantes ML (Scikit-Learn)

* **Prédiction Box-office :** Modèle de régression (Random Forest / Gradient Boosting) entraîné sur les données TMDB, prenant en compte le budget, le genre et la saisonnalité.

---

## 📖 Fonctionnalités principales

### Pour la Production & l'Écriture

* **Script Assistant :** Analyse technique et interrogation de scénarios PDF.
* **Market Intelligence :** Évaluation de l'originalité d'un pitch par rapport aux sorties récentes et à venir (2024-2026).
* **Logistics Automation :** Génération automatique de feuilles de dépouillement technique (lieux, personnages, besoins FX) à partir d'un résumé.
* **Casting & Creative :** Suggestions d'acteurs basées sur les archétypes de personnages et rédaction de scripts de bande-annonce.

### Pour l'Analyse Spectateur

* **Assistant Intelligent :** Interface conversationnelle pour la recommandation et l'identification de genres cinématographiques.
* **Générateur de Critiques :** Extraction de points forts/faibles et verdict technique automatisé.

---

## ⚙️ Installation et Déploiement

### Prérequis

* Python 3.9+
* Clé API Mistral (via `console.mistral.ai`)
* Clé API Tavily (pour l'analyse de marché web)

### Configuration

1. Cloner le dépôt et installer les dépendances :

```bash
pip install -r requirements.txt

```

2. Configurer les variables d'environnement (`.env`) :

```ini
MISTRAL_API_KEY=votre_cle
TAVILY_API_KEY=votre_cle

```

3. Lancer l'interface :

```bash
streamlit run app.py

```

---

## 📁 Structure des modules

* `app.py` : Point d'entrée Streamlit et gestion de l'état de la session.
* `llm_utils.py` : Logique des chaînes LangChain, schémas Pydantic et outils RAG.
* `ml_utils.py` : Entraînement, comparaison de modèles et fonctions d'inférence statistique.
* `config.py` : Gestion centralisée des variables de chemin et hyperparamètres des modèles.
* `finetune_prep.py` / `finetune_run.py` : Utilitaires pour l'entraînement cloud de modèles Mistral personnalisés.

---

## 📈 Pipeline de données ML

Le modèle prédictif traite les données brutes de `tmdb_5000_movies.csv` en appliquant :

1. Un encodage des genres via `LabelEncoder`.
2. Une transformation logarithmique des revenus (`np.log1p`) pour stabiliser la variance.
3. Une sélection automatisée du meilleur modèle basée sur le score .
