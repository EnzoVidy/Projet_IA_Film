import streamlit as st
import pandas as pd
from llm_utils import (
    recommander_films,
    genre_depuis_synopsis,
    generer_critique,
    generer_synopsis,
    generer_casting,
    corriger_script,
    generer_bande_annonce
)
# Importation de la nouvelle partie ML
from ml_utils import predict_box_office, train_model

st.set_page_config(page_title="Filmind", layout="wide")
st.title("🎬 Filmind – Suite IA Complète")

# Sidebar
profil = st.sidebar.selectbox(
    "Qui êtes-vous ?",
    ["Consommateur de films", "Producteur / Créateur de films", "(Admin)"]
)

menu_options = []

if profil == "Consommateur de films":
    menu_options = [
        "Recommandations de films",
        "Identifier le genre d’un film",
        "Générer une critique"
    ]
elif profil == "Producteur / Créateur de films":
    menu_options = [
        "💰 Prédiction Box-Office (IA Prédictive)", # NOUVEAU
        "Générer un synopsis",
        "Trouver un casting",
        "Corriger un script / logline",
        "Créer une bande-annonce (texte)"
    ]
elif profil == "(Admin)":
    menu_options = ["Gestion des Modèles"]

menu = st.sidebar.selectbox("Choisissez une fonctionnalité :", menu_options)

# ---------------------------------------------------------
# CONSOMMATEUR
# ---------------------------------------------------------

if profil == "Consommateur de films":
    if menu == "Recommandations de films":
        st.subheader("⭐ Recommander des films à partir de tes goûts")
        films_aimes = st.text_area("Liste quelques œuvres que tu aimes :")
        if st.button("Générer recommandations"):
            with st.spinner("L'IA réfléchit..."):
                recommandations = recommander_films(films_aimes)
            st.write("### 🎯 Suggestions :")
            st.write(recommandations)

    elif menu == "Identifier le genre d’un film":
        st.subheader("🎭 Identifier le genre d’un film")
        synopsis = st.text_area("Entre le synopsis du film :")
        if st.button("Détecter le genre"):
            with st.spinner("Analyse en cours..."):
                genre = genre_depuis_synopsis(synopsis)
            st.success(f"Genre détecté : **{genre}**")

    elif menu == "Générer une critique":
        st.subheader("📝 Générer une critique")
        titre = st.text_input("Nom du film :")
        description = st.text_area("Résumé / quelques infos sur le film :")
        if st.button("Créer critique"):
            with st.spinner("Rédaction en cours..."):
                critique = generer_critique(titre, description)
            st.write("### 📄 Critique générée :")
            st.write(critique)

# ---------------------------------------------------------
# PRODUCTEUR (AVEC IA PRÉDICTIVE)
# ---------------------------------------------------------

elif profil == "Producteur / Créateur de films":

    # --- NOUVELLE FONCTIONNALITÉ PRÉDICTIVE ---
    if menu == "💰 Prédiction Box-Office (IA Prédictive)":
        st.subheader("📊 Estimer le succès commercial (Machine Learning)")
        st.info("Ce module utilise un modèle Random Forest entraîné sur 5000 films historiques.")
        
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Budget du film ($)", min_value=1000, value=1000000, step=10000)
        with col2:
            runtime = st.number_input("Durée (minutes)", min_value=10, value=90, step=1)
            
        if st.button("Prédire les revenus"):
            prediction = predict_box_office(budget, runtime)
            if prediction:
                st.metric(label="Revenus Estimés (Box Office)", value=f"{prediction:,.2f} $")
                
                # Petit calcul de ROI pour l'affichage
                roi = ((prediction - budget) / budget) * 100
                color = "green" if roi > 0 else "red"
                st.markdown(f"ROI estimé : <span style='color:{color}'>**{roi:.1f}%**</span>", unsafe_allow_html=True)
            else:
                st.error("Le modèle n'est pas encore entraîné. Allez dans le menu ' ou vérifiez le fichier CSV.")

    elif menu == "Générer un synopsis":
        st.subheader("📚 Générer un synopsis")
        titre = st.text_input("Titre du film :")
        if st.button("Générer synopsis"):
            with st.spinner("Création..."):
                synopsis = generer_synopsis(titre)
            st.write("### 📘 Synopsis proposé :")
            st.write(synopsis)

    elif menu == "Trouver un casting":
        st.subheader("👥 Trouver un casting adapté")
        synopsis = st.text_area("Synopsis du film :")
        if st.button("Générer casting"):
            with st.spinner("Recherche des acteurs..."):
                casting = generer_casting(synopsis)
            st.write("### 🎭 Casting proposé :")
            st.write(casting)

    elif menu == "Corriger un script / logline":
        st.subheader("🛠 Correction de script")
        texte = st.text_area("Colle ici ton texte :")
        if st.button("Corriger"):
            with st.spinner("Correction..."):
                correction = corriger_script(texte)
            st.write("### ✔ Correction :")
            st.write(correction)

    elif menu == "Créer une bande-annonce (texte)":
        st.subheader("🎤 Générer une bande-annonce (Script)")
        synopsis = st.text_area("Synopsis du film :")
        if st.button("Créer bande-annonce"):
            with st.spinner("Écriture du script..."):
                ba = generer_bande_annonce(synopsis)
            st.write("### 🎬 Bande-annonce :")
            st.write(ba)

# ---------------------------------------------------------
# DATA SCIENTIST (ADMIN)
# ---------------------------------------------------------

elif profil == "(Admin)":
    st.subheader("⚙️ Administration des modèles IA")
    
    st.write("### 1. Modèle Prédictif (Random Forest)")
    st.write("Permet de prédire le Box-Office.")
    if st.button("Entraîner le modèle prédictif (Reload)"):
        with st.spinner("Entraînement du modèle sur tmdb_5000_movies.csv..."):
            res = train_model()
        st.success(res)
        
    st.divider()
    
    st.write("### 2. Fine-Tuning Mistral (LLM)")
    st.write("Générer les données JSONL pour le fine-tuning sur la plateforme Mistral.")
    if st.button("Générer fichier JSONL"):
        try:
            from finetune_prep import prepare_mistral_finetuning_data
            prepare_mistral_finetuning_data()
            st.success("Fichier 'mistral_finetune.jsonl' généré à la racine !")
            with open("mistral_finetune.jsonl", "r") as f:
                st.download_button("Télécharger le JSONL", f, "mistral_finetune.jsonl")
        except ImportError:
            st.error("Le fichier finetune_prep.py est introuvable.")