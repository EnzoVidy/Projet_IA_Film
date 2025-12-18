import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from config import MISTRAL_MODEL, LLM_TEMPERATURE
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
# FONCTIONS SPECTATEUR (Agent & Classique)
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
    prompt = ChatPromptTemplate.from_template("Expert cinéma. L'utilisateur aime : {films_aimes}. Recommande 5 films.")
    return (prompt | llm_struct).invoke({"films_aimes": films_aimes}).films

def genre_depuis_synopsis(synopsis):
    llm_struct = model.with_structured_output(GenreResponse)
    prompt = ChatPromptTemplate.from_template("Analyse ce synopsis et extrais le genre : {synopsis}")
    return (prompt | llm_struct).invoke({"synopsis": synopsis}).genre

def generer_critique(titre, description):
    prompt = ChatPromptTemplate.from_template("Écris une critique pro de 15 lignes pour {titre}. Infos : {description}")
    return (prompt | model | parser).invoke({"titre": titre, "description": description})

def generer_critique_json(titre, description):
    llm_struct = model.with_structured_output(CritiqueStructuree)
    prompt = ChatPromptTemplate.from_template("Génère une critique structurée pro pour {titre}. Infos : {description}")
    return (prompt | llm_struct).invoke({"titre": titre, "description": description})

# ==========================================
# FONCTIONS PRODUCTEUR
# ==========================================

def generer_synopsis(titre):
    prompt = ChatPromptTemplate.from_template("Génère un synopsis de 25-70 mots pour : {titre}")
    return (prompt | model | parser).invoke({"titre": titre})

def generer_casting(synopsis):
    llm_struct = model.with_structured_output(CastingResponse)
    prompt = ChatPromptTemplate.from_template("Propose un casting (Principal, Antagoniste, etc.) pour : {synopsis}")
    return (prompt | llm_struct).invoke({"synopsis": synopsis}).roles

def generer_bande_annonce(synopsis):
    prompt = ChatPromptTemplate.from_template("""Tu es une voix-off pro. {synopsis}. 
    Retourne UNIQUEMENT la VOIX-OFF. Structure : Intro mystère, Tension, Conflit, Final percutant.""")
    return (prompt | model | parser).invoke({"synopsis": synopsis})

def analyser_concurrence_web(pitch):
    prompt_keywords = ChatPromptTemplate.from_template("Extrais mots clés en anglais pour : {pitch}")
    mots_cles = (prompt_keywords | model | parser).invoke({"pitch": pitch})
    search = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)
    try:
        results = search.invoke(f"movie plot similar to {mots_cles} 2024 2025")
        context_web = str(results)
    except:
        context_web = "Aucun résultat trouvé."
    llm_struct = model.with_structured_output(RapportConcurrence)
    prompt = ChatPromptTemplate.from_template("Analyse la concurrence. Pitch: {pitch}. Web: {context_web}")
    return (prompt | llm_struct).invoke({"pitch": pitch, "context_web": context_web})

# ==========================================
# RAG & TECHNIQUE
# ==========================================

def creer_vecteur_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def interroger_scenario(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Script-doctor. Contexte: {context}"),
        ("human", "{question}")
    ])
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | parser)
    return chain.invoke(question)

def generer_depouillement(texte_sequence):
    llm_struct = model.with_structured_output(Depouillement)
    prompt = ChatPromptTemplate.from_template("Dépouillement technique (scènes, lieux, FX) de : {texte}")
    return (prompt | llm_struct).invoke({"texte": texte_sequence}).scenes

def generer_fiche_personnage_json(description):
    llm_struct = model.with_structured_output(CharacterProfile)
    prompt = ChatPromptTemplate.from_template("Fiche personnage complète pour : {description}")
    return (prompt | llm_struct).invoke({"description": description})

def extraire_parametres_box_office(texte):
    llm_struct = model.with_structured_output(BoxOfficeParams)
    prompt = ChatPromptTemplate.from_template("Extrais Budget ($) et Durée (min) de : {texte}")
    return (prompt | llm_struct).invoke({"texte": texte})