"""
Configuration centralisée de l'application.
Aucun modèle IA n'est chargé ici : on stocke juste des CONSTANTES.
"""

# --- Modèles Hugging Face ---
WHISPER_MODEL_NAME = "openai/whisper-small"

# CLIP fait de la classification "zero-shot" : on lui donne des labels
# candidats en langage naturel, sans avoir besoin de le fine-tuner.
# C'est plus pertinent qu'un ViT classique (entraîné sur ImageNet,
# donc incapable de détecter un "colis endommagé" nativement).
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"

VISION_CANDIDATE_LABELS = [
    "a damaged product",
    "a broken package",
    "an intact product in good condition",
    "a product that does not match the description",
]

# --- Fichiers ---
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp3"}
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE_MB = 15

# --- RAG ---
RAG_DATA_PATH = "app/rag_data/cgv_faq.json"
RAG_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_TOP_K = 1
