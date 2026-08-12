"""
Point d'entrée de l'application.

Pattern Singleton via lifespan (identique à ton service Whisper) :
tous les modèles lourds (Whisper, CLIP, sentence-transformers) + les
embeddings du corpus RAG sont chargés UNE SEULE FOIS au démarrage du
serveur, et stockés dans app.state. Chaque requête réutilise ces objets
déjà en mémoire -> pas de rechargement, pas de saturation RAM.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from app.core.config import WHISPER_MODEL_NAME, VISION_MODEL_NAME, RAG_EMBEDDING_MODEL
from app.services.rag_service import load_corpus, build_corpus_embeddings
from app.routes.support import router as support_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Démarrage : chargement unique de tous les modèles ---
    # Stockés dans app.state -> accessibles depuis n'importe quelle route
    # via `request.app.state`, sans import circulaire avec main.py.
    print("Chargement du modèle ASR (Whisper)...")
    app.state.asr_model = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_NAME)

    print("Chargement du modèle Vision (CLIP zero-shot)...")
    app.state.vision_model = pipeline("zero-shot-image-classification", model=VISION_MODEL_NAME)

    print("Chargement du modèle d'embeddings (RAG)...")
    app.state.embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL)

    print("Encodage du corpus CGV/FAQ...")
    app.state.rag_corpus = load_corpus()
    app.state.rag_embeddings = build_corpus_embeddings(app.state.embedding_model, app.state.rag_corpus)

    print("Tous les modèles sont chargés. API prête.")
    yield

    # --- Arrêt : libération propre ---
    app.state.asr_model = None
    app.state.vision_model = None
    app.state.embedding_model = None
    print("Modèles déchargés, arrêt propre.")


app = FastAPI(
    title="Support Ticket AI API",
    description="Ingestion multimodale (audio/image/texte) de réclamations client "
                 "avec transcription ASR, diagnostic visuel et recherche RAG dans les CGV.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(support_router)


@app.get("/", tags=["Santé"])
def root():
    return {"status": "ok", "message": "Support Ticket AI API is running"}
