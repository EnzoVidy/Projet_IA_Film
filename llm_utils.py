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
Format : 10 lignes, style professionnel.
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

Propose un casting d’acteurs adaptés :
- Rôle principal
- Antagoniste
- Rôle secondaire
- Cameo

Explique brièvement pourquoi chaque acteur est pertinent.
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

Objectifs :
- Fluidité
- Structure
- Clarté
- Ton cinématographique

Retourne uniquement la version corrigée.
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
Tu es une voix-off de bande-annonce hollywoodienne.
À partir du synopsis suivant : {synopsis}

Génère une bande-annonce dramatique, format voix off :
- Introduction mystérieuse
- Tension qui monte
- Présentation du conflit
- Phrase choc finale

Style : "In a world where..."
        """
    )
    chain = prompt | model | parser
    return chain.invoke({"synopsis": synopsis})