# 🎬 Filmind – Suite IA & Cinéma

**Filmind** est une plateforme intelligente conçue pour accompagner l'industrie cinématographique. En combinant la puissance des **LLM (Mistral AI via LangChain)** et du **Machine Learning (Scikit-Learn)**, l'application offre des outils d'analyse et de création tant pour les cinéphiles que pour les professionnels de la production.

---

## ✨ Fonctionnalités

### 🍿 Espace Spectateur

* **Assistant IA (Agent) :** Un point d'entrée unique capable de comprendre l'intention de l'utilisateur (recommander, critiquer ou identifier un genre).
* **Système de Recommandation :** Suggestions personnalisées basées sur vos goûts cinématographiques.
* **Identification de Genre :** Analyse automatique du genre à partir d'un simple synopsis.
* **Générateur de Critiques :** Rédaction de critiques structurées avec notation, points forts et points faibles.

### 💼 Espace Producteur & Créateur

* **Prédiction Box-Office :** Estimation des revenus générés à l'aide d'un modèle de régression entraîné sur les données historiques.
* **Éclaireur de Marché :** Analyse de la concurrence en temps réel via une recherche web (Tavily) pour évaluer l'originalité d'un projet.
* **Assistant Scénario (RAG) :** Téléchargez votre script PDF et posez des questions complexes sur l'intrigue ou les personnages.
* **Dépouillement Technique :** Transformation automatique d'un récit en tableau de bord logistique (lieux, personnages, besoins spéciaux).
* **Outils Créatifs :** Génération de *loglines*, suggestions de casting idéal, création de scripts de bande-annonce et fiches personnages au format JSON.

### ⚙️ Administration & ML Ops

* **Entraînement de Modèles :** Comparaison de modèles (Random Forest, Gradient Boosting, etc.) pour optimiser les prédictions.
* **Fine-Tuning Mistral :** Pipeline complet pour préparer les données et lancer des jobs de fine-tuning sur le cloud Mistral AI.

---

## 🛠️ Stack Technique

* **Interface :** [Streamlit](https://streamlit.io/)
* **Intelligence Artificielle :** [Mistral AI](https://mistral.ai/), [LangChain](https://www.langchain.com/)
* **Machine Learning :** Pandas, Scikit-Learn, Joblib
* **RAG (Retrieval Augmented Generation) :** FAISS, HuggingFace Embeddings
* **Recherche Web :** Tavily API

---

## 🚀 Installation

1. **Cloner le dépôt :**
```bash
git clone https://github.com/votre-compte/filmind.git
cd filmind

```


2. **Installer les dépendances :**
```bash
pip install -r requirements.txt

```


3. **Configurer les variables d'environnement :**
Créez un fichier `.env` à la racine du projet :
```env
MISTRAL_API_KEY=votre_cle_mistral
TAVILY_API_KEY=votre_cle_tavily

```


4. **Vérifier la configuration :**
```bash
python check_deps.py

```


5. **Lancer l'application :**
```bash
streamlit run app.py

```



---

## 📁 Structure du Projet

| Fichier | Rôle |
| --- | --- |
| `app.py` | Interface utilisateur principale (Streamlit). |
| `llm_utils.py` | Logique des agents, du RAG et des sorties structurées. |
| `ml_utils.py` | Pipeline d'entraînement et de prédiction Box-Office. |
| `config.py` | Configuration centralisée (chemins, modèles, paramètres). |
| `finetune_prep.py` | Préparation des datasets au format `.jsonl`. |
| `finetune_run.py` | Communication avec l'API Mistral pour le fine-tuning. |

---

## 📊 Modèle Prédictif

Le modèle de prédiction du Box-Office utilise les données de **TMDB** pour corréler le budget, la durée, la popularité et le genre avec les revenus mondiaux. Le système sélectionne automatiquement le modèle le plus performant (par défaut **Random Forest**) lors de la phase d'administration.
