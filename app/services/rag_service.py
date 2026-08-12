"""
Service RAG (Retrieval-Augmented Generation — partie "Retrieval" uniquement).

Principe :
1. Au démarrage (via main.py/lifespan), on encode UNE FOIS tout le corpus
   CGV/FAQ en vecteurs (embeddings) et on les garde en mémoire (numpy array).
2. À chaque requête, on encode la question du client et on calcule la
   similarité cosinus avec chaque règle du corpus.
3. On retourne la règle la plus proche (top-1).

Pourquoi pas de base de données vectorielle (FAISS, Chroma...) ici ?
Pour un corpus de quelques dizaines de règles, une simple comparaison
numpy est largement suffisante et évite une dépendance/infra inutile.
Si le corpus grossissait (milliers de documents), FAISS deviendrait
pertinent pour la recherche approximative rapide (ANN).
"""

import json
import numpy as np
from app.core.config import RAG_DATA_PATH, RAG_TOP_K


def load_corpus(path: str = RAG_DATA_PATH) -> list[dict]:
    """Charge le corpus CGV/FAQ depuis le fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus_embeddings(embedding_model, corpus: list[dict]) -> np.ndarray:
    """
    Encode tout le corpus en une seule fois (appelé au démarrage de l'app,
    jamais à chaque requête -> c'est la clé de l'optimisation mémoire/CPU).
    """
    textes = [doc["texte"] for doc in corpus]
    embeddings = embedding_model.encode(textes, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def _cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """
    Les vecteurs sont déjà normalisés (normalize_embeddings=True),
    donc la similarité cosinus se réduit à un simple produit scalaire.
    """
    return corpus_vecs @ query_vec


def search_rule(embedding_model, corpus: list[dict], corpus_embeddings: np.ndarray,
                 query_text: str, top_k: int = RAG_TOP_K) -> dict | None:
    """
    Cherche la règle CGV/FAQ la plus pertinente pour un texte donné.

    Returns:
        dict avec "texte", "statut_associe", "score" -- ou None si le
        corpus est vide ou le texte de requête est vide.
    """
    if not query_text or not query_text.strip() or len(corpus) == 0:
        return None

    query_vec = embedding_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    scores = _cosine_similarity(query_vec, corpus_embeddings)

    best_idx = int(np.argmax(scores))
    best_doc = corpus[best_idx]

    return {
        "regle_applicable": best_doc["texte"],
        "statut_associe": best_doc["statut_associe"],
        "score": round(float(scores[best_idx]), 3),
    }
