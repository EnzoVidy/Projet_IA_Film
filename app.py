import os
import streamlit as st
import pandas as pd
from llm_utils import (
    recommander_films,
    genre_depuis_synopsis,
    generer_critique,
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
# Importation de la nouvelle partie ML
# On importe la nouvelle fonction qui compare les modèles
from ml_utils import predict_box_office, train_and_compare_models
from config import TEMP_DIR

st.set_page_config(page_title="Filmind", layout="wide")
st.title("🎬 Filmind – Suite IA Complète")

# Sidebar
profil = st.sidebar.selectbox(
    "Qui êtes-vous ?",
    ["Spectateur", "Producteur / Créateur de films", "(Admin)"]
)

menu_options = []

if profil == "Spectateur":
    menu_options = [
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
# SPECTATEUR
# ---------------------------------------------------------

if profil == "Spectateur":
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
# PRODUCTEUR
# ---------------------------------------------------------

elif profil == "Producteur / Créateur de films":

    # --- NOUVELLE FONCTIONNALITÉ PRÉDICTIVE ---
    if menu == "Prédiction Box-Office (IA Prédictive)":
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

    elif menu == "Générer un synopsis":
        st.subheader("📚 Générer un synopsis")
        titre = st.text_input("Titre du film :")
        if st.button("Générer synopsis"):
            with st.spinner("Création..."):
                synopsis = generer_synopsis(titre)
            st.write("### Synopsis proposé :")
            st.write(synopsis)

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
        st.subheader("🧬 Générateur de Personnage (Structured Output)")
        st.info("Cette fonctionnalité génère une réponse strictement formatée en JSON.")
        
        idee = st.text_input("Idée du personnage (ex: Un détective cyborg dépressif) :")
        
        if st.button("Générer JSON"):
            with st.spinner("Construction de la structure de données..."):
                profil_perso = generer_fiche_personnage_json(idee)
                
            # Affichage visuel (User Friendly)
            st.write(f"### 👤 {profil_perso.nom}")
            st.write(f"**Archétype :** {profil_perso.archetype}")
            st.write(f"**Backstory :** {profil_perso.backstory}")
            
            # Affichage du JSON brut (Pour le prof / Dev)
            st.divider()
            st.write("### 💻 Sortie JSON Brute (Backend)")
            # On utilise .model_dump_json() de Pydantic pour avoir le string JSON
            st.json(profil_perso.model_dump_json())

# ---------------------------------------------------------
# DATA SCIENTIST (ADMIN)
# ---------------------------------------------------------

elif profil == "(Admin)":
    st.subheader("⚙️ Administration des modèles IA")
    
    # --- PARTIE 1 : ML COMPARATIF ---
    st.write("### 1. Comparateur de Modèles (Auto-ML)")
    st.write("Entraîne plusieurs modèles et sélectionne le meilleur pour les prédictions.")
    
    if st.button("Lancer l'entraînement et la comparaison"):
        with st.spinner("Entraînement : Linear Reg vs Random Forest vs Gradient Boosting..."):
            df_results, best_name = train_and_compare_models()
        
        if isinstance(df_results, str): # Cas d'erreur
            st.error(df_results)
        else:
            st.success(f"Entraînement terminé ! Le meilleur modèle est : **{best_name}**")
            
            # Affichage du tableau comparatif
            st.write("#### 📊 Tableau des performances")
            st.dataframe(df_results.style.highlight_max(axis=0, subset=["R2 Score"]), use_container_width=True)
            
            st.markdown("""
            **Critères d'évaluation :**
            * $R^2$ (Score) : Plus c'est proche de 1, plus le modèle "explique" bien la variance.
            * **MAE (Mean Absolute Error)** : L'erreur moyenne en dollars.
            """)
            
            # Graphique simple
            st.write("#### Comparaison du Score R2")
            st.bar_chart(df_results.set_index("Modèle")["R2 Score"])

    st.divider()
    
    # --- PARTIE 2 : LLM FINE-TUNING ---
    st.write("### 2. Fine-Tuning Mistral (LLM)")
    
    st.info("Étape A : Préparer les données")
    if st.button("Générer fichier JSONL"):
        try:
            from finetune_prep import prepare_mistral_finetuning_data
            prepare_mistral_finetuning_data()
            st.success("Fichier 'mistral_finetune.jsonl' généré à la racine !")
            with open("mistral_finetune.jsonl", "r") as f:
                st.download_button("Télécharger le JSONL (Backup)", f, "mistral_finetune.jsonl")
        except ImportError:
            st.error("Le fichier finetune_prep.py est introuvable.")

    st.info("Étape B : Envoyer chez Mistral AI")
    if st.button("🚀 LANCER LE FINE-TUNING (CLOUD)"):
        # On vérifie d'abord que le fichier existe
        if not os.path.exists("mistral_finetune.jsonl"):
            st.error("Génère d'abord le fichier JSONL ci-dessus !")
        else:
            try:
                # Import dynamique pour éviter les erreurs si mistralai n'est pas installé
                from finetune_run import launch_mistral_finetuning
                
                with st.spinner("Contact de l'API Mistral (Upload + Start Job)..."):
                    message, job_id = launch_mistral_finetuning()
                
                if job_id:
                    st.success(message)
                    st.balloons()
                    st.markdown(f"""
                        **Que faire maintenant ?**
                        1. Va sur [La Console Mistral](https://console.mistral.ai/fine-tuning) pour suivre l'avancée.
                        2. Quand le statut est `SUCCESS`, copie le nom du nouveau modèle.
                        3. Colle-le dans `config.py` à la place de `MISTRAL_MODEL`.
                    """)
                else:
                    st.error(message)
            except ImportError:
                st.error("Le module 'mistralai' n'est pas installé. Fais 'pip install mistralai'.")
            except Exception as e:
                st.error(f"Erreur script : {e}")