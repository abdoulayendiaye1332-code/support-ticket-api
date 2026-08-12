"""
Schémas Pydantic : ils définissent le "contrat" de sortie de l'API.
Avantage direct : FastAPI génère automatiquement la doc Swagger (/docs)
à partir de ces classes, sans effort supplémentaire.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    """
    Statuts proposés pour orienter l'équipe support.
    Doit correspondre EXACTEMENT aux valeurs "statut_associe" du corpus
    CGV (app/rag_data/cgv_faq.json) -- sinon la conversion en Enum plante.
    """
    REMBOURSABLE = "Remboursable"
    A_VERIFIER = "À vérifier"
    ECHANGE_GRATUIT = "Échange gratuit"
    EXPEDITION_PIECE = "Expédition de pièce"
    NON_REMBOURSABLE_RETARD_MINEUR = "Non remboursable - Retard mineur"
    DEDOMMAGEMENT_10 = "Dédommagement 10%"
    REMBOURSABLE_COLIS_PERDU = "Remboursable - Colis perdu"
    REFUSE = "Refusé"
    EN_ATTENTE_JUSTIFICATIFS = "En attente de justificatifs"


class VisionDiagnostic(BaseModel):
    label: str = Field(..., description="Diagnostic retenu (ex: 'a damaged product')")
    confidence: float = Field(..., description="Score de confiance du modèle (0 à 1)")


class RagResult(BaseModel):
    regle_applicable: Optional[str] = Field(
        None, description="Extrait de la CGV/FAQ jugé le plus pertinent"
    )
    score: Optional[float] = Field(
        None, description="Score de similarité avec la requête"
    )


class SupportTicketResponse(BaseModel):
    transcription: Optional[str] = Field(
        None, description="Texte transcrit depuis l'audio (si fourni)"
    )
    diagnostic_image: Optional[VisionDiagnostic] = Field(
        None, description="Résultat de l'analyse visuelle (si image fournie)"
    )
    texte_analyse: Optional[str] = Field(
        None, description="Texte final utilisé pour interroger le RAG "
                           "(transcription, ou description, ou les deux)"
    )
    rag: Optional[RagResult] = Field(
        None, description="Règle interne trouvée par le RAG"
    )
    statut_propose: TicketStatus = Field(
        ..., description="Statut suggéré pour orienter le ticket"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transcription": "le colis est arrivé cassé",
                "diagnostic_image": {"label": "a damaged product", "confidence": 0.87},
                "texte_analyse": "le colis est arrivé cassé",
                "rag": {
                    "regle_applicable": "Si le client signale un produit cassé... "
                                         "Statut associé : Remboursable.",
                    "score": 0.79,
                },
                "statut_propose": "Remboursable",
            }
        }
