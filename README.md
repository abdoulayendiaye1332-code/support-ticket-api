# Support Ticket AI API

API multimodale d'automatisation du support client pour SmartHelp (e-commerce).
Elle ingère un vocal, une photo et/ou un texte de réclamation client, et retourne
un diagnostic structuré (transcription, analyse visuelle, règle CGV applicable,
statut proposé) pour orienter instantanément l'équipe support.

## Contexte

Le service client reçoit les réclamations via messagerie (notes vocales, photos).
L'analyse manuelle (écoute, lecture des CGV) prend du temps. Cette API automatise
la première analyse pour aiguiller le ticket vers le bon traitement.

## Architecture

```
support-ticket-api/
├── app/
│   ├── main.py              # Démarrage FastAPI + chargement singleton des modèles (lifespan)
│   ├── core/
│   │   └── config.py         # Constantes (noms de modèles, seuils, chemins)
│   ├── routes/
│   │   └── support.py        # Endpoint POST /support-ticket (orchestration HTTP uniquement)
│   ├── schemas/
│   │   └── ticket.py         # Contrats Pydantic (validation + doc Swagger)
│   ├── services/
│   │   ├── asr_service.py    # Transcription audio (Whisper)
│   │   ├── vision_service.py # Diagnostic image (CLIP zero-shot)
│   │   └── rag_service.py    # Recherche de règle CGV (similarité cosinus)
│   └── rag_data/
│       └── cgv_faq.json      # Base de connaissances interne (CGV SmartHelp, 9 règles)
├── requirements.txt
├── .gitignore
└── README.md
```

Le découpage suit une séparation stricte des responsabilités : les **routes**
gèrent le HTTP (validation de fichiers, nettoyage, erreurs), les **services**
contiennent la logique IA pure (aucune dépendance à FastAPI), et les **schemas**
définissent le contrat de données. Cela permet de remplacer un modèle IA sans
toucher au reste du code.

## Choix techniques

### Transcription audio — Whisper (`openai/whisper-small`)
Modèle encoder-decoder de référence pour l'ASR, multilingue, bon compromis
taille/performance pour un usage sans GPU dédié.

### Analyse visuelle — CLIP (`openai/clip-vit-base-patch32`) en zero-shot
Plutôt qu'un ViT classique (entraîné sur ImageNet, donc incapable de reconnaître
un "produit endommagé" nativement), CLIP permet une classification **zero-shot** :
on lui fournit une liste de labels en langage naturel (`"a damaged product"`,
`"an intact product"`, etc.) sans avoir besoin d'un dataset annoté ni de
fine-tuning. Choix pragmatique pour un projet sans corpus d'images labellisées.

### RAG — sentence-transformers + numpy (pas de FAISS)
Le corpus CGV (9 règles, voir `app/rag_data/cgv_faq.json`) est encodé une seule
fois au démarrage avec `paraphrase-multilingual-MiniLM-L12-v2` (modèle
multilingue léger, adapté au français). La recherche se fait par similarité
cosinus, calculée directement en numpy (produit scalaire sur vecteurs
normalisés). Pour un corpus de cette taille (quelques dizaines de règles),
une base vectorielle dédiée (FAISS, Chroma) aurait été une dépendance inutile ;
elle deviendrait pertinente à partir de plusieurs milliers de documents.

### Optimisation mémoire — Singleton via `lifespan`
Les 3 modèles (Whisper, CLIP, sentence-transformers) et les embeddings du
corpus RAG sont chargés **une seule fois** au démarrage du serveur, via le
gestionnaire de contexte `lifespan` de FastAPI, et stockés dans `app.state`.
Chaque requête réutilise ces objets déjà en mémoire : aucun rechargement,
donc pas de saturation RAM ni de latence supplémentaire par requête.

## Installation

```bash
git clone https://github.com/abdoulayendiaye1332-code/support-ticket-api.git
cd support-ticket-api

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Torch en CPU-only (évite plusieurs Go de dépendances CUDA inutiles
# sur une machine sans GPU dédié)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## Lancement

```bash
uvicorn app.main:app --reload
```

Le premier démarrage télécharge les modèles depuis Hugging Face (peut prendre
plusieurs minutes). Une fois prêt, le terminal affiche :

```
Tous les modèles sont chargés. API prête.
```

Documentation interactive Swagger : http://127.0.0.1:8000/docs

## Utilisation de l'API

### `POST /support-ticket`

Requête `multipart/form-data` avec au moins un des trois champs :

| Champ | Type | Description |
|---|---|---|
| `audio` | fichier (.mp3, .wav) | Note vocale du client |
| `image` | fichier (.png, .jpg) | Photo du produit |
| `description` | texte | Description écrite optionnelle |

**Exemple avec `curl` (texte seul) :**

```bash
curl -X POST http://127.0.0.1:8000/support-ticket \
  -F "description=mon colis est arrivé avec la boîte fissurée"
```

**Exemple de réponse :**

```json
{
  "transcription": null,
  "diagnostic_image": null,
  "texte_analyse": "mon colis est arrivé avec la boîte fissurée",
  "rag": {
    "regle_applicable": "Si le client signale un produit cassé, fissuré ou endommagé, et fournit une photo probante de l'article dans un délai de 48 heures suivant la réception, le dossier est éligible à un remboursement intégral ou à un renvoi gratuit.",
    "score": 0.269
  },
  "statut_propose": "Remboursable"
}
```

**Exemple avec audio et/ou image :**

```bash
curl -X POST http://127.0.0.1:8000/support-ticket \
  -F "audio=@note_vocale.mp3" \
  -F "image=@produit.jpg"
```

### Statuts possibles

`Remboursable`, `À vérifier`, `Échange gratuit`, `Expédition de pièce`,
`Non remboursable - Retard mineur`, `Dédommagement 10%`,
`Remboursable - Colis perdu`, `Refusé`, `En attente de justificatifs`.

Le statut est directement dérivé de la règle CGV la plus proche trouvée par le
RAG. En l'absence de correspondance ou de texte à analyser, le statut par
défaut est `À vérifier` — aucune décision automatique n'est prise sans base
documentaire.

## Gestion des erreurs et robustesse

- **Validation des types de fichiers** : rejet (`415`) si le type MIME de
  l'audio ou de l'image n'est pas dans la liste autorisée
  (`app/core/config.py`).
- **Limite de taille** : rejet (`413`) au-delà de 15 Mo par fichier.
- **Nettoyage garanti des fichiers temporaires** : chaque fichier uploadé est
  écrit dans un fichier temporaire (`tempfile`), supprimé dans un bloc
  `finally`, donc même en cas d'erreur pendant le traitement.
- **Aucune fuite de stack trace** : toute exception interne est interceptée et
  retournée sous forme de `500` avec un message générique, jamais la trace
  Python brute.

## État des tests

-  Cas texte seul (`description`) : validé, RAG retourne la bonne règle avec
  un statut cohérent.
-  Cas image seule : validé, `diagnostic_image` génère un résultat exploitable.
-  Cas audio : en cours de validation.
-  Cas combiné (audio + image + description) : à tester.

## Gestion de projet

- **Kanban** : https://trello.com/b/MleCGm8s/support-ticket-api — colonnes Backlog / In Progress / Review / Done.
- **Git Flow** : `main` (stable) / `develop` (intégration) / `feature/...`
  (une branche par fonctionnalité, fusionnée dans `develop` via merge `--no-ff`).

## Auteur

Projet réalisé en solo par Adoulaye Ndiaye, dans le cadre du parcours de formation
développement web & IA.