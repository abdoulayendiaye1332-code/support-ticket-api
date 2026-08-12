"""
Service Vision.
On utilise CLIP en "zero-shot image classification" : on ne l'entraîne pas,
on lui donne juste une liste de labels candidats en langage naturel
(cf VISION_CANDIDATE_LABELS dans core/config.py). Le modèle renvoie un score
de correspondance pour chaque label. C'est le choix pragmatique quand on n'a
pas de dataset annoté de "produits endommagés" pour fine-tuner un ViT classique.
"""

from PIL import Image
from app.core.config import VISION_CANDIDATE_LABELS


def analyze_image(vision_pipeline, image_path: str) -> dict:
    """
    Analyse une image produit et retourne le label le plus probable.

    Args:
        vision_pipeline: pipeline zero-shot-image-classification (singleton)
        image_path: chemin vers l'image temporaire sur disque

    Returns:
        dict {"label": str, "confidence": float}
    """
    image = Image.open(image_path).convert("RGB")
    results = vision_pipeline(image, candidate_labels=VISION_CANDIDATE_LABELS)

    # results est une liste triée par score décroissant, ex:
    # [{"label": "a damaged product", "score": 0.87}, ...]
    best = results[0]
    return {"label": best["label"], "confidence": round(best["score"], 3)}
