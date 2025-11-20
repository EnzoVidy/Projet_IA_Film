import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

model = ChatMistralAI(
    model="mistral-medium-latest",
    temperature=0.7,
    api_key=api_key
)

parser = StrOutputParser()

# -------------------------------
# 1. Recommandations
# -------------------------------

def recommander_films(films):
    prompt = ChatPromptTemplate.from_template(
        """
        Tu es un expert cinéma spécialisé en recommandations personnalisées.
        À partir des films suivants : {films}

        Ta mission :
        - Trouve 10 films REELS (aucune invention)
        - Triés par similarité thématique + ambiance + style
        - Justification courte (max 10 mots)
        - Format minimaliste, pas d’introduction ni conclusion
        - Aucun bonus, aucun commentaire hors liste

        Format :

        1. Titre (année) — justification
        2. ...
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"films": films})

# -------------------------------
# 2. Genre depuis synopsis
# -------------------------------

def genre_depuis_synopsis(synopsis):
    prompt = ChatPromptTemplate.from_template(
        """
Analyse ce synopsis : {synopsis}
Donne uniquement le genre principal du film (un seul mot).
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"synopsis": synopsis})

# -------------------------------
# 3. Générer critique
# -------------------------------

def generer_critique(titre, description):
    prompt = ChatPromptTemplate.from_template(
        """
        Écris une critique professionnelle du film "{titre}".
        Informations : {description}

        Longueur : 15 lignes. Ton professionnel mais fluide.
        Concis, objectif, avec essentiellement les informations fournies par l'utilisateur.
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"titre": titre, "description": description})

# -------------------------------
# 4. Générer synopsis
# -------------------------------

def generer_synopsis(titre):
    prompt = ChatPromptTemplate.from_template(
        """
        Génère un synopsis original pour un film intitulé : "{titre}".
        Format : strictement 25 à 70 mots, style professionnel.
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"titre": titre})

# -------------------------------
# 5. Générer casting
# -------------------------------

def generer_casting(synopsis):
    prompt = ChatPromptTemplate.from_template(
        """
        À partir du synopsis suivant : {synopsis}

        Génère un casting concis et professionnel.
        CONTRAINTES STRICTES :
        - Pas de bonus, pas d'analyse de film, pas d'esthétique visuelle.
        - Ne fournis QUE le casting.
        - Pour chaque rôle : 1 acteur + justification courte (max 2 lignes).
        - Format imposé :

        Rôle principal : Nom de l’acteur
        Justification : ...

        Antagoniste : Nom de l’acteur
        Justification : ...

        Secondaire : Nom de l’acteur
        Justification : ...

        ...

        Cameo : Nom de l’acteur
        Justification : ...

        Ne dépasse jamais 15 lignes au total.
        Respecte le caractère du synopsis, même s'il est fantastique ou historique.
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"synopsis": synopsis})

# -------------------------------
# 6. Correction script
# -------------------------------

def corriger_script(texte):
    prompt = ChatPromptTemplate.from_template(
        """
        Corrige le texte suivant :
        {texte}

        Règles strictes :
        - Ne modifie PAS l'histoire.
        - Ne change PAS le sens.
        - Améliore : vocabulaire, clarté, structure, style cinématographique léger.

        Retourne UNIQUEMENT la version corrigée, concise.
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"texte": texte})

# -------------------------------
# 7. Bande annonce
# -------------------------------

def generer_bande_annonce(synopsis):
    prompt = ChatPromptTemplate.from_template(
        """
        Tu es une voix-off professionnelle spécialisée dans les bandes-annonces hollywoodiennes.

        À partir du synopsis suivant :
        {synopsis}

        Produit une bande-annonce TEXTUELLE, au format VOIX-OFF uniquement.

        Contraintes :
        - Garde un ton cinématographique mais concis.
        - N'ajoute pas trop de nouveaux éléments qui n’existent pas dans le synopsis.
        - Structure obligatoire (mais à ne pas montrer) :
            1. Introduction mystérieuse
            2. Tension qui monte / présentation du contexte
            3. Présentation du conflit
            4. Phrase finale percutante
        - Pas de scènes très longues ou inventées : reste sobre et évocateur.
        - Pas de *storytelling* détaillé, pas de dialogues inventés.
        - Pas de notes, pas d’explications, pas de musique/ambiance décrite en détail.

        Retourne UNIQUEMENT la bande-annonce de la VOIX-OFF, donc seulement ce qui sera dit par la voix-off.
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"synopsis": synopsis})