import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from config import MISTRAL_MODEL, MISTRAL_MODEL_FINETUNE, LLM_TEMPERATURE
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

model = ChatMistralAI(
    model=MISTRAL_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=api_key
)

modelFinetuned = ChatMistralAI(
    model=MISTRAL_MODEL_FINETUNE,
    temperature=LLM_TEMPERATURE,
    api_key=api_key
)

parser = StrOutputParser()

# ==========================================
# MODÈLES DE DONNÉES (STRUCTURED OUTPUT)
# ==========================================

class FilmRecommande(BaseModel):
    titre: str = Field(description="Le titre du film")
    annee: str = Field(description="L'année de sortie")
    justification: str = Field(description="Pourquoi ce film correspond à la demande (court)")

class ReponseRecommandations(BaseModel):
    films: List[FilmRecommande]

class GenreResponse(BaseModel):
    genre: str = Field(description="Le genre principal du film (ex: Drame, SF, Comédie)")

class CritiqueStructuree(BaseModel):
    note_etoiles: int = Field(description="Note globale de 1 à 5")
    points_forts: List[str] = Field(description="Liste des 3 points les plus positifs")
    points_faibles: List[str] = Field(description="Liste des 3 points les plus négatifs")
    analyse_technique: str = Field(description="Avis sur la mise en scène ou le jeu d'acteur (1 phrase)")
    verdict: str = Field(description="Le mot de la fin")

class Role(BaseModel):
    nom_role: str = Field(description="Le type de rôle (Principal, Antagoniste, etc.)")
    acteur_suggere: str = Field(description="Nom de l'acteur ou l'actrice")
    raison: str = Field(description="Pourquoi cet acteur correspond (physique, jeu, etc.)")

class CastingResponse(BaseModel):
    roles: List[Role]

class RapportConcurrence(BaseModel):
    concurrents_identifies: List[str] = Field(description="Liste des titres de films trouvés ayant un synopsis très proche")
    analyse_courte: str = Field(description="Analyse critique : Est-ce du déjà-vu ? (Max 2 phrases)")
    score_originalite: int = Field(description="Score sur 100")
    film_menace: str = Field(description="Le titre du film le plus dangereux (ou 'Aucun')")

class SceneInfo(BaseModel):
    numero_scene: int = Field(description="Numéro estimé de la scène")
    lieu: str = Field(description="Lieu de l'action")
    moment: str = Field(description="Jour, Nuit, Aube, Crépuscule")
    personnages: List[str] = Field(description="Liste des personnages présents")
    besoins_speciaux: str = Field(description="FX, animaux, véhicules, cascade...")

class Depouillement(BaseModel):
    scenes: List[SceneInfo]

class CharacterProfile(BaseModel):
    nom: str = Field(description="Nom complet du personnage")
    archetype: str = Field(description="L'archétype (ex: Le Mentor, L'Élu)")
    motivations: List[str] = Field(description="Liste de 3 motivations principales")
    backstory: str = Field(description="Court résumé de son passé")
    points_faibles: List[str] = Field(description="Liste des faiblesses psychologiques")

class BoxOfficeParams(BaseModel):
    budget: float = Field(description="Budget en $")
    runtime: float = Field(description="Durée en minutes")

class IntentRouter(BaseModel):
    action: Literal["recommander", "identifier_genre", "generer_critique"] = Field(description="L'action demandée")
    donnee_principale: str = Field(description="Contenu extrait (films, synopsis ou titre)")
    infos_supplemementaires: Optional[str] = Field(description="Notes additionnelles")

# ==========================================
# FONCTIONS SPECTATEUR
# ==========================================

def router_demande_spectateur(requete_utilisateur):
    llm_router = model.with_structured_output(IntentRouter)
    prompt = ChatPromptTemplate.from_template(
        "Analyse la requête utilisateur : \"{requete}\". "
        "Choisis entre : recommander, identifier_genre, generer_critique."
    )
    return (prompt | llm_router).invoke({"requete": requete_utilisateur})

def recommander_films(films_aimes):
    llm_struct = model.with_structured_output(ReponseRecommandations)
    
    prompt = ChatPromptTemplate.from_template(
        """
        Tu es un expert cinéma. L'utilisateur aime : {films_aimes}.
        Recommande 5 films pertinents. Sois précis sur les années.
        """
    )
    chain = prompt | llm_struct
    reponse = chain.invoke({"films_aimes": films_aimes})
    return reponse.films

def genre_depuis_synopsis(synopsis):
    llm_struct = model.with_structured_output(GenreResponse)
    prompt = ChatPromptTemplate.from_template(
        "Analyse ce synopsis et extrais le genre principal : {synopsis}"
    )
    chain = prompt | llm_struct
    reponse = chain.invoke({"synopsis": synopsis})
    return reponse.genre

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

def generer_critique_json(titre, description):
    llm_struct = model.with_structured_output(CritiqueStructuree)
    prompt = ChatPromptTemplate.from_template("Génère une critique structurée pro pour {titre}. Infos : {description}")
    return (prompt | llm_struct).invoke({"titre": titre, "description": description})

# ==========================================
# FONCTIONS PRODUCTEUR
# ==========================================

def generer_logline(titre):
    prompt_generation = ChatPromptTemplate.from_template(
        """Based on the movie title '{titre}', generate a compelling, professional logline.
        Style: Dramatic, cinematic, concise (under 50 words).
        """
    )
    
    logline_anglais = (prompt_generation | modelFinetuned | parser).invoke({"titre": titre})

    prompt_traduction = ChatPromptTemplate.from_template(
        """Tu es un adaptateur de scénario expert.
        Traduis cette logline en FRANÇAIS.
        Garde le ton dramatique et le style "cinéma".
        
        logline originale : "{texte}"
        
        Retourne uniquement la traduction française.
        """
    )
    
    res_final = (prompt_traduction | model | parser).invoke({"texte": logline_anglais})
    return res_final


def generer_casting(synopsis):
    llm_struct = model.with_structured_output(CastingResponse)
    
    prompt = ChatPromptTemplate.from_template(
        """
        À partir du synopsis : {synopsis}
        Propose un casting idéal (Principal, Secondaire, Antagoniste, Cameo).
        """
    )
    chain = prompt | llm_struct
    reponse = chain.invoke({"synopsis": synopsis})
    return reponse.roles

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

def analyser_concurrence_web(pitch):
    prompt_keywords = ChatPromptTemplate.from_template(
        """
        Extrais les concepts clés de ce pitch de film pour une recherche Google.
        Traduis-les en anglais pour de meilleurs résultats.
        Format attendu : juste les mots clés séparés par des espaces.
        
        Pitch : "{pitch}"
        """
    )
    chain_keywords = prompt_keywords | model | parser
    mots_cles = chain_keywords.invoke({"pitch": pitch})

    requete_optimisee = f"movie plot similar to {mots_cles} released in 2023 2024 2025 or upcoming 2026"
    
    print(f"Recherche Tavily lancée : {requete_optimisee}") 

    search = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)
    
    try:
        results_raw = search.invoke(requete_optimisee)
        context_web = str(results_raw)
    except Exception as e:
        context_web = f"Erreur de recherche : {e}"

    llm_struct = model.with_structured_output(RapportConcurrence)    
    prompt_analyse = ChatPromptTemplate.from_template(
        """
        Tu es un analyste de marché cinéma impitoyable.
        
        TON PITCH CIBLE : "{pitch}"
        
        RÉSULTATS DE LA RECHERCHE WEB (Films récents/à venir) :
        {context_web}
        
        Instructions :
        1. Compare le pitch cible UNIQUEMENT avec les films trouvés dans les résultats web.
        2. Si tu trouves un film dans les résultats avec une histoire très similaire (plus de 70% de ressemblance), le score d'originalité doit être BAS (< 40).
        3. Ignore les vieux films classiques (années 80, 90, 2000) sauf s'ils sont dans les résultats de recherche. Concentre-toi sur la nouveauté.
        4. Remplis le rapport structuré.
        """
    )
    
    chain = prompt_analyse | llm_struct
    return chain.invoke({"pitch": pitch, "context_web": context_web})

# ==========================================
# RAG & TECHNIQUE
# ==========================================

def creer_vecteur_store(pdf_path):
    """Indexe un PDF pour le RAG"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def interroger_scenario(vectorstore, question):
    """Pose une question au PDF"""
    retriever = vectorstore.as_retriever()

    prompt_answer = ChatPromptTemplate.from_messages([
        ("system", 
         "Tu es un script-doctor assistant. Utilise le contexte ci-dessous :\n\n{context}\n\n"
         "Réponds précisément à la question."),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever,
            "question": lambda x: x["question"],
        }
        | prompt_answer
        | model
        | StrOutputParser()
    )
    return rag_chain.invoke({"question": question})

def generer_depouillement(texte_sequence):
    llm_struct = model.with_structured_output(Depouillement)
    prompt = ChatPromptTemplate.from_template(
        """
        Analyse le texte suivant (synopsis détaillé ou séquence de script).
        Réalise un "dépouillement" technique pour la production.
        Découpe l'action en scènes distinctes.
        
        Texte : {texte}
        """
    )
    chain = prompt | llm_struct
    return chain.invoke({"texte": texte_sequence}).scenes

def generer_fiche_personnage_json(description):
    llm_struct = model.with_structured_output(CharacterProfile)
    prompt = ChatPromptTemplate.from_template("Fiche personnage complète pour : {description}")
    return (prompt | llm_struct).invoke({"description": description})

def extraire_parametres_box_office(texte):
    llm_struct = model.with_structured_output(BoxOfficeParams)
    prompt = ChatPromptTemplate.from_template("Extrais Budget ($) et Durée (min) de : {texte}")
    return (prompt | llm_struct).invoke({"texte": texte})