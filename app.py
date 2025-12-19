import os
import streamlit as st
import pandas as pd
from llm_utils import (
    recommander_films,
    genre_depuis_synopsis,
    generer_critique,
    generer_critique_json,
    router_demande_spectateur,
    generer_logline,
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
        "Assistant IA (Agent)",
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
        "Générer une logline",
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

    if menu == "Assistant IA (Agent)":
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

    # --- MENUS CLASSIQUES ---
    if menu == "Recommandations de films":
        st.subheader("⭐ Recommander des films à partir de tes goûts")
        films_aimes = st.text_area("Liste quelques œuvres que tu aimes :")
        
        if st.button("Générer recommandations"):
            with st.spinner("L'IA réfléchit..."):
                recommandations = recommander_films(films_aimes)
            
            st.write("### Suggestions :")
            for film in recommandations:
                with st.expander(f"🎬 {film.titre} ({film.annee})"):
                    st.write(f"**Pourquoi ?** {film.justification}")


    elif menu == "Identifier le genre d’un film":
        st.subheader("🎭 Identifier le genre d’un film")
        synopsis = st.text_area("Entre le synopsis du film :")
        if st.button("Détecter le genre"):
            with st.spinner("Analyse en cours..."):
                genre = genre_depuis_synopsis(synopsis)
            st.markdown(f"### Genre détecté : :blue-background[{genre}]")


    elif menu == "Générer une critique":
        st.subheader("📝 Générer une critique")
        titre = st.text_input("Nom du film :")
        description = st.text_area("Résumé / quelques infos sur le film :")
        if st.button("Créer critique"):
            with st.spinner("Rédaction en cours..."):
                critique = generer_critique(titre, description)
            st.write("### Critique générée :")
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
        pitch = st.text_area("Ton idée de film / Pitch :")
        if st.button("Lancer l'enquête"):
            with st.spinner("Scan du web pour les films en production..."):
                rapport = analyser_concurrence_web(pitch)
            col1, col2, col3 = st.columns(3)
            col1.metric("Score Originalité", f"{rapport.score_originalite}/100")
            col2.metric("Menace Principale", rapport.film_menace)
            col3.metric("Concurrents", len(rapport.concurrents_identifies))
            st.warning(f"**Analyse :** {rapport.analyse_courte}")
            with st.expander("Voir les concurrents identifiés"):
                for f in rapport.concurrents_identifies:
                    st.write(f"- {f}")

    elif menu == "Assistant Scénario":
        st.subheader("🤖 Discuter avec ton script (RAG)")
        uploaded_file = st.file_uploader("Upload ton scénario (PDF)", type="pdf")
        if uploaded_file:
            temp_path = TEMP_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if "vectorstore" not in st.session_state:
                with st.spinner("Indexation du document..."):
                    st.session_state.vectorstore = creer_vecteur_store(str(temp_path))
                st.success("Scénario lu et mémorisé !")
            question = st.text_input("Pose une question sur ton scénario :")
            if st.button("Demander"):
                with st.spinner("Recherche dans le PDF..."):
                    reponse = interroger_scenario(st.session_state.vectorstore, question)
                st.write(reponse)

    elif menu == "Dépouillement Technique":
        st.subheader("🎬 Générer une feuille de service")
        st.info("Transforme un récit en tableau pour la logistique.")
        texte = st.text_area("Colle une séquence ou un résumé détaillé :")
        if st.button("Générer Tableau"):
            with st.spinner("Extraction des données..."):
                scenes = generer_depouillement(texte)
            data = []
            for s in scenes:
                data.append({
                    "Scène N°": s.numero_scene,
                    "Lieu": s.lieu,
                    "Moment": s.moment,
                    "Personnages": ", ".join(s.personnages),
                    "Besoins Spéciaux": s.besoins_speciaux
                })
            df = pd.DataFrame(data)
            st.table(df)

    elif menu == "Trouver un casting":
        st.subheader("👥 Trouver un casting adapté")
        synopsis = st.text_area("Synopsis du film :")
        
        if st.button("Générer casting"):
            with st.spinner("Recherche des acteurs..."):
                casting_list = generer_casting(synopsis)
            
            st.write("### Casting proposé :")
            for role in casting_list:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**{role.nom_role}**")
                    st.info(f"👤 {role.acteur_suggere}")
                with c2:
                    st.markdown("*Pourquoi ?*")
                    st.write(role.raison)
                st.divider()

    elif menu == "Créer une bande-annonce (texte)":
        st.subheader("🎤 Générer une bande-annonce (Script)")
        synopsis = st.text_area("Synopsis du film :")
        if st.button("Créer bande-annonce"):
            with st.spinner("Écriture du script..."):
                ba = generer_bande_annonce(synopsis)
            st.write("### Bande-annonce :")
            st.write(ba)

    elif menu == "Générer Fiche Perso (JSON)":
        st.subheader("🧬 Structured Output : Personnage")
        idee = st.text_input("Concept (ex: Pirate de l'espace)")
        if st.button("Générer JSON"):
            perso = generer_fiche_personnage_json(idee)
            st.json(perso.model_dump())

    elif menu == "Générer une logline":
        t = st.text_input("Titre :")
        if st.button("Créer"): st.write(generer_logline(t))

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