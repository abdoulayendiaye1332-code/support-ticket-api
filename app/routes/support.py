"""
Route unique d'ingestion des tickets support.

Rôle de ce fichier : ORCHESTRER uniquement.
- Valider les fichiers reçus
- Gérer les fichiers temporaires (création + nettoyage GARANTI)
- Appeler les services (asr, vision, rag)
- Construire la réponse Pydantic

Toute la logique IA est déléguée aux services. Cette route ne "sait pas"
comment fonctionne Whisper ou CLIP -- elle sait juste QUAND les appeler.
"""

import os
import tempfile
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException

from app.core.config import ALLOWED_AUDIO_TYPES, ALLOWED_IMAGE_TYPES, MAX_FILE_SIZE_MB
from app.schemas.ticket import SupportTicketResponse, VisionDiagnostic, RagResult, TicketStatus
from app.services.asr_service import transcribe_audio
from app.services.vision_service import analyze_image
from app.services.rag_service import search_rule

router = APIRouter(tags=["Support Ticket"])


def _normalize_upload(file: UploadFile | str | None) -> UploadFile | None:
    """
    L'interface Swagger envoie parfois une chaîne vide "" pour un champ
    fichier optionnel non rempli, au lieu de ne rien envoyer. On normalise
    ça en None, ainsi qu'un UploadFile sans nom de fichier (cas limite).
    """
    if file is None or isinstance(file, str):
        return None
    if not file.filename:
        return None
    return file


async def _save_upload_to_temp(upload: UploadFile, suffix: str) -> str:
    """
    Sauvegarde un UploadFile sur disque dans un fichier temporaire et
    retourne son chemin. On vérifie la taille au passage.
    """
    content = await upload.read()

    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux ({size_mb:.1f} Mo, max {MAX_FILE_SIZE_MB} Mo).",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def _determine_status(rag_match: dict | None) -> TicketStatus:
    """
    Le statut proposé est directement dérivé de la règle CGV trouvée
    (champ "statut_associe" dans le corpus). Si aucune règle pertinente
    n'est trouvée (score bas ou pas de texte), on renvoie "À vérifier"
    par défaut -> on ne prend jamais de décision automatique risquée
    sans base documentaire.
    """
    if rag_match is None:
        return TicketStatus.A_VERIFIER
    return TicketStatus(rag_match["statut_associe"])


@router.post("/support-ticket", response_model=SupportTicketResponse)
async def create_support_ticket(
    request: Request,
    audio: UploadFile | str | None = File(None, description="Note vocale du client (.mp3, .wav)"),
    image: UploadFile | str | None = File(None, description="Photo du produit (.png, .jpg)"),
    description: str | None = Form(None, description="Texte descriptif optionnel du client"),
):
    audio = _normalize_upload(audio)
    image = _normalize_upload(image)

    if audio is None and image is None and not description:
        raise HTTPException(
            status_code=400,
            detail="Au moins un des trois éléments (audio, image, description) est requis.",
        )

    temp_files: list[str] = []
    transcription = None
    diagnostic_image = None

    try:
        # --- 1. Audio -> transcription ---
        if audio is not None:
            if audio.content_type not in ALLOWED_AUDIO_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail=f"Type audio non supporté : {audio.content_type}",
                )
            audio_path = await _save_upload_to_temp(audio, suffix=".wav")
            temp_files.append(audio_path)
            transcription = transcribe_audio(request.app.state.asr_model, audio_path)

        # --- 2. Image -> diagnostic visuel ---
        if image is not None:
            if image.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail=f"Type image non supporté : {image.content_type}",
                )
            image_path = await _save_upload_to_temp(image, suffix=".jpg")
            temp_files.append(image_path)
            diagnostic_image = analyze_image(request.app.state.vision_model, image_path)

        # --- 3. Texte final utilisé pour interroger le RAG ---
        texte_analyse = transcription or description or ""

        # --- 4. RAG ---
        rag_match = search_rule(
            embedding_model=request.app.state.embedding_model,
            corpus=request.app.state.rag_corpus,
            corpus_embeddings=request.app.state.rag_embeddings,
            query_text=texte_analyse,
        )

        # --- 5. Construction de la réponse ---
        return SupportTicketResponse(
            transcription=transcription,
            diagnostic_image=VisionDiagnostic(**diagnostic_image) if diagnostic_image else None,
            texte_analyse=texte_analyse or None,
            rag=RagResult(
                regle_applicable=rag_match["regle_applicable"],
                score=rag_match["score"],
            ) if rag_match else None,
            statut_propose=_determine_status(rag_match),
        )

    except HTTPException:
        raise
    except Exception as e:
        # On ne laisse jamais fuir une stack trace brute au client.
        raise HTTPException(status_code=500, detail=f"Erreur interne de traitement : {str(e)}")

    finally:
        # Nettoyage GARANTI des fichiers temporaires, même en cas d'erreur.
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
