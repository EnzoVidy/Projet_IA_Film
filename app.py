import streamlit as st
from llm_utils import (
    recommander_films,
    genre_depuis_synopsis,
    generer_critique,
    generer_synopsis,
    generer_casting,
    corriger_script,
    generer_bande_annonce
)

st.set_page_config(page_title="Filmind", layout="wide")
st.title("🎬 Filmind – Analyse & Génération pour Films")

# Choix du profil utilisateur
profil = st.sidebar.selectbox(
    "Qui êtes-vous ?",
    ["Consommateur de films", "Producteur / Créateur de films"]
)

# Définition des fonctionnalités selon le profil
if profil == "Consommateur de films":
    menu_options = [
        "Recommandations de films",
        "Identifier le genre d’un film",
        "Générer une critique"
    ]
elif profil == "Producteur / Créateur de films":
    menu_options = [
        "Générer un synopsis",
        "Trouver un casting",
        "Corriger un script / logline",
        "Créer une bande-annonce (texte)"
    ]

# Menu de fonctionnalités
menu = st.sidebar.selectbox("Choisissez une fonctionnalité :", menu_options)

# ---------------------------------------------------------
# AFFICHAGE DES FONCTIONNALITÉS
# ---------------------------------------------------------

# Consommateurs
if profil == "Consommateur de films":

    if menu == "Recommandations de films":
        st.subheader("⭐ Recommander des films à partir de tes goûts")
        films_aimes = st.text_area("Liste quelques œuvres que tu aimes :")
        if st.button("Générer recommandations"):
            recommandations = recommander_films(films_aimes)
            st.write("### 🎯 Suggestions :")
            st.write(recommandations)

    elif menu == "Identifier le genre d’un film":
        st.subheader("🎭 Identifier le genre d’un film")
        synopsis = st.text_area("Entre le synopsis du film :")
        if st.button("Détecter le genre"):
            genre = genre_depuis_synopsis(synopsis)
            st.success(f"Genre détecté : **{genre}**")

    elif menu == "Générer une critique":
        st.subheader("📝 Générer une critique")
        titre = st.text_input("Nom du film :")
        description = st.text_area("Résumé / quelques infos sur le film :")
        if st.button("Créer critique"):
            critique = generer_critique(titre, description)
            st.write("### 📄 Critique générée :")
            st.write(critique)

# Producteurs
elif profil == "Producteur / Créateur de films":

    if menu == "Générer un synopsis":
        st.subheader("📚 Générer un synopsis à partir du titre du film")
        titre = st.text_input("Titre du film :")
        if st.button("Générer synopsis"):
            synopsis = generer_synopsis(titre)
            st.write("### 📘 Synopsis proposé :")
            st.write(synopsis)

    elif menu == "Trouver un casting":
        st.subheader("👥 Trouver un casting adapté")
        synopsis = st.text_area("Synopsis du film :")
        if st.button("Générer casting"):
            casting = generer_casting(synopsis)
            st.write("### 🎭 Casting proposé :")
            st.write(casting)

    elif menu == "Corriger un script / logline":
        st.subheader("🛠 Correction de script / logline")
        texte = st.text_area("Colle ici ton texte à corriger :")
        if st.button("Corriger"):
            correction = corriger_script(texte)
            st.write("### ✔ Correction :")
            st.write(correction)

    elif menu == "Créer une bande-annonce (texte)":
        st.subheader("🎤 Générer une bande-annonce (texte type voix-off)")
        synopsis = st.text_area("Synopsis du film :")
        if st.button("Créer bande-annonce"):
            ba = generer_bande_annonce(synopsis)
            st.write("### 🎬 Bande-annonce :")
            st.write(ba)
