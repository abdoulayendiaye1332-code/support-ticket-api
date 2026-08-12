"""
Service ASR (Automatic Speech Recognition).
Le modèle est chargé UNE SEULE FOIS par main.py (lifespan) et injecté ici
via ml_models. Cette fonction ne fait QUE transcrire : pas de logique HTTP,
pas de gestion de fichiers temporaires (ça, c'est le rôle de la route).
"""

from transformers import pipeline


def transcribe_audio(asr_pipeline: "pipeline", audio_path: str) -> str:
    """
    Transcrit un fichier audio en texte.

    Args:
        asr_pipeline: pipeline Whisper déjà chargé (singleton depuis main.py)
        audio_path: chemin vers le fichier audio temporaire sur disque

    Returns:
        Le texte transcrit (str, potentiellement vide si audio inaudible)
    """
    result = asr_pipeline(audio_path)
    return result["text"].strip()
