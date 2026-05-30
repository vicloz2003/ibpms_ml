"""
Policy classification via multilingual sentence embeddings (Deep Learning / NLP).

Uses a pretrained transformer (paraphrase-multilingual-MiniLM-L12-v2) — a BERT-family
neural network — to embed the client's request and each policy's (description + tags),
then ranks policies by cosine similarity. No labeled training data required: the model
is pretrained and works in Spanish out of the box.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from models.schemas import PolicyInput, PolicyMatch, ClassifyResponse

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cosine score below which we consider the match too weak to act on automatically.
CONFIDENCE_THRESHOLD = 0.30


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Lazily load the model once (heavy: ~470MB). Cached for the process lifetime."""
    return SentenceTransformer(MODEL_NAME)


def warm_up() -> None:
    """Force model load at startup so the first request isn't slow."""
    _model().encode(["warm up"])


def _policy_document(p: PolicyInput) -> str:
    """Build the text that represents a policy for embedding: name + description + tags."""
    parts = [p.name or "", p.description or ""]
    if p.tags:
        parts.append(" ".join(p.tags))
    return ". ".join(part for part in parts if part).strip()


def classify(text: str, policies: List[PolicyInput]) -> ClassifyResponse:
    if not policies:
        return ClassifyResponse(
            policyId=None, policyName=None, confidence=0.0, confident=False,
            alternatives=[], message="No hay políticas activas para clasificar.",
        )

    model = _model()
    query_vec = model.encode(text, normalize_embeddings=True)
    policy_docs = [_policy_document(p) for p in policies]
    policy_vecs = model.encode(policy_docs, normalize_embeddings=True)

    # Cosine similarity (vectors already normalized → dot product)
    scores = np.dot(policy_vecs, query_vec)

    ranked = sorted(
        (
            PolicyMatch(policyId=p.id, policyName=p.name, score=round(float(s), 4))
            for p, s in zip(policies, scores)
        ),
        key=lambda m: m.score,
        reverse=True,
    )

    top = ranked[0]
    confident = top.score >= CONFIDENCE_THRESHOLD
    message = (
        f"Política recomendada: {top.policyName}"
        if confident
        else "No estoy seguro de qué trámite necesitas. ¿Puedes describirlo con más detalle?"
    )

    return ClassifyResponse(
        policyId=top.policyId if confident else None,
        policyName=top.policyName if confident else None,
        confidence=top.score,
        confident=confident,
        alternatives=ranked[:3],
        message=message,
    )
