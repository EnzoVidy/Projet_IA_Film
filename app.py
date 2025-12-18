import os
import streamlit as st
import pandas as pd
from llm_utils import (
    recommander_films,
    genre_depuis_synopsis,
    generer_critique,
    generer_critique_json, # Ajouté
    router_demande_spectateur, # Ajouté
    generer_synopsis,
    generer_casting,
    generer_bande_annonce,
    analyser_concurrence_web,
    creer_vecteur_store,
    interroger_scenario,
    generer_depouillement,
    generer_fiche_personnage_json,
    extraire_parametres_box_office
)
from ml_utils import predict_box_office, train_and_compare_models
from config import TEMP_DIR
# https://console.mistral.ai/build/finetuned-models/jobs
st.set_page_config(page_title="Filmind – IA & Cinéma", layout="wide")
st.title("🎬 Filmind – Suite IA Complète")

# Sidebar - Profils
profil = st.sidebar.selectbox(
    "Qui êtes-vous ?",
    ["Spectateur", "Producteur / Créateur de films", "(Admin)"]
)

menu_options = []

if profil == "Spectateur":
    menu_options = [
        "🤖 Assistant IA (Agent)", # Nouvelle option centralisée
        "Recommandations de films",
        "Identifier le genre d’un film",
        "Générer une critique"
    ]
elif profil == "Producteur / Créateur de films":
    menu_options = [
        "Prédiction Box-Office (IA Prédictive)",
        "Éclaireur de Marché",
        "Assistant Scénario",
        "Dépouillement Technique",
        "Générer un synopsis",
        "Trouver un casting",
        "Créer une bande-annonce (texte)",
        "Générer Fiche Perso (JSON)"
    ]
elif profil == "(Admin)":
    menu_options = ["Gestion des Modèles"]

menu = st.sidebar.selectbox("Choisissez une fonctionnalité :", menu_options)

# ---------------------------------------------------------
# SECTION SPECTATEUR
# ---------------------------------------------------------
if profil == "Spectateur":

    # --- NOUVELLE FONCTIONNALITÉ AGENT ---
    if menu == "🤖 Assistant IA (Agent)":
        st.subheader("🕵️ Assistant Intelligent (Tout-en-un)")
        st.write("Posez votre question naturellement. L'IA choisira l'outil adapté.")
        
        user_query = st.text_input("Ex: 'Conseille moi des films comme Matrix' ou 'Fais la critique de Batman'")
        
        if st.button("Lancer l'analyse") and user_query:
            with st.spinner("L'agent réfléchit..."):
                # 1. On route l'intention
                decision = router_demande_spectateur(user_query)
                
                # 2. On exécute selon l'intention
                if decision.action == "recommander":
                    st.info(f"Action détectée : Recommandation (Sujet : {decision.donnee_principale})")
                    recs = recommander_films(decision.donnee_principale)
                    for film in recs:
                        with st.expander(f"🎬 {film.titre} ({film.annee})"):
                            st.write(film.justification)

                elif decision.action == "identifier_genre":
                    st.info(f"Action détectée : Analyse de genre")
                    genre = genre_depuis_synopsis(decision.donnee_principale)
                    st.success(f"Le genre détecté est : **{genre}**")

                elif decision.action == "generer_critique":
                    st.info(f"Action détectée : Critique (Film : {decision.donnee_principale})")
                    critique = generer_critique_json(decision.donnee_principale, decision.infos_supplemementaires or "")
                    
                    col1, col2 = st.columns([1, 2])
                    col1.metric("Note", f"{critique.note_etoiles}/5")
                    col2.write(f"**Verdict :** {critique.verdict}")
                    
                    c1, c2 = st.columns(2)
                    c1.success("**Points Forts**\n\n" + "\n".join([f"- {p}" for p in critique.points_forts]))
                    c2.error("**Points Faibles**\n\n" + "\n".join([f"- {p}" for p in critique.points_faibles]))

    # --- MENUS CLASSIQUES (GARDÉS) ---
    elif menu == "Recommandations de films":
        st.subheader("⭐ Recommander des films")
        films_aimes = st.text_area("Liste quelques œuvres que tu aimes :")
        if st.button("Générer"):
            recommandations = recommander_films(films_aimes)
            for film in recommandations:
                st.write(f"🎬 **{film.titre}** : {film.justification}")

    elif menu == "Identifier le genre d’un film":
        st.subheader("🎭 Identifier le genre")
        synopsis = st.text_area("Entre le synopsis :")
        if st.button("Détecter"):
            genre = genre_depuis_synopsis(synopsis)
            st.markdown(f"Genre : :blue-background[{genre}]")

    elif menu == "Générer une critique":
        st.subheader("📝 Générer une critique")
        titre = st.text_input("Nom du film :")
        desc = st.text_area("Infos :")
        if st.button("Créer"):
            critique = generer_critique(titre, desc)
            st.write(critique)

# ---------------------------------------------------------
# SECTION PRODUCTEUR
# ---------------------------------------------------------
elif profil == "Producteur / Créateur de films":
    
    if menu == "Prédiction Box-Office (IA Prédictive)":
        st.subheader("📊 Estimer le succès commercial")
        col1, col2 = st.columns(2)
        budget = col1.number_input("Budget ($)", min_value=0, value=1000000)
        runtime = col2.number_input("Durée (min)", min_value=0, value=90)
        
        if st.button("Prédire"):
            res = predict_box_office(budget, runtime)
            if res:
                st.metric("Revenus Estimés", f"{res:,.2f} $")

    elif menu == "Éclaireur de Marché":
        st.subheader("🌐 Analyse de la concurrence")
        pitch = st.text_area("Ton pitch :")
        if st.button("Lancer l'enquête"):
            rapport = analyser_concurrence_web(pitch)
            st.metric("Score Originalité", f"{rapport.score_originalite}/100")
            st.write(f"**Analyse :** {rapport.analyse_courte}")

    elif menu == "Assistant Scénario":
        st.subheader("🤖 RAG : Discuter avec le script")
        file = st.file_uploader("Upload PDF", type="pdf")
        if file:
            temp_path = TEMP_DIR / file.name
            with open(temp_path, "wb") as f: f.write(file.getbuffer())
            if "vs" not in st.session_state:
                st.session_state.vs = creer_vecteur_store(str(temp_path))
            q = st.text_input("Question sur le script :")
            if st.button("Demander"):
                st.write(interroger_scenario(st.session_state.vs, q))

    elif menu == "Dépouillement Technique":
        st.subheader("🎬 Feuille de service")
        txt = st.text_area("Texte de la séquence :")
        if st.button("Générer"):
            scenes = generer_depouillement(txt)
            st.table(pd.DataFrame([s.model_dump() for s in scenes]))

    elif menu == "Générer Fiche Perso (JSON)":
        st.subheader("🧬 Structured Output : Personnage")
        idee = st.text_input("Concept (ex: Pirate de l'espace)")
        if st.button("Générer JSON"):
            perso = generer_fiche_personnage_json(idee)
            st.json(perso.model_dump())

    # Autres menus (synopsis, casting, BA) à garder selon ta structure initiale...
    elif menu == "Générer un synopsis":
        t = st.text_input("Titre :")
        if st.button("Créer"): st.write(generer_synopsis(t))

# ---------------------------------------------------------
# SECTION ADMIN
# ---------------------------------------------------------
elif profil == "(Admin)":
    st.subheader("⚙️ Administration")
    
    if st.button("Lancer l'entraînement ML (Comparaison)"):
        df_res, best = train_and_compare_models()
        st.dataframe(df_res)
        st.success(f"Meilleur modèle : {best}")

    st.divider()
    
    st.write("### Fine-Tuning Mistral")
    if st.button("1. Préparer JSONL"):
        from finetune_prep import prepare_mistral_finetuning_data
        prepare_mistral_finetuning_data()
        st.success("Fichier prêt !")
        
    if st.button("2. Lancer Job Cloud"):
        from finetune_run import launch_mistral_finetuning
        msg, jid = launch_mistral_finetuning()
        st.write(msg)