"""Logique métier : évaluer une soumission, enregistrer une revue humaine, statistiques."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import kobo
from .config import get_settings
from .db import Beneficiary, Review, utcnow
from .ml import load_model
from .schema import normalize_features


def ingest_submission(session: Session, payload: dict, source: str) -> tuple[Beneficiary, bool]:
    """Évalue une soumission Kobo et l'enregistre. Retourne (ménage, créé?).

    Une soumission renvoyée par Kobo (nouvel essai, synchronisation, ou
    modification de la soumission) met à jour la ligne existante au lieu de
    créer un doublon. Si les réponses ont changé après une revue humaine, la
    revue est annulée et le ménage repasse "en attente".
    """
    key, uuid = kobo.submission_key(payload)
    flat = kobo.flatten(payload)
    features = normalize_features(flat)  # lève ValidationError si une réponse est invalide
    settings = get_settings()
    result = load_model(settings.model_dir).predict(features, review_band=settings.review_band)

    ben = session.scalar(select(Beneficiary).where(Beneficiary.kobo_key == key))
    created = ben is None
    if created:
        ben = Beneficiary(kobo_key=key, source=source, statut_revue="en_attente")
        session.add(ben)

    ben.kobo_uuid = uuid
    ben.submitted_at = kobo.submission_time(payload)
    ben.received_at = utcnow()
    ben.raw_payload = payload
    for field, value in kobo.identity(flat).items():
        setattr(ben, field, value)
    if not created and ben.features == features:
        # seules les informations d'identification ont pu changer : la décision reste valable
        session.commit()
        return ben, False

    ben.features = features
    ben.score = result["score"]
    ben.threshold = result["threshold"]
    ben.model_decision = result["eligible"]
    ben.a_verifier = result["a_verifier"]
    ben.base_value = result["base_value"]
    ben.explanation = result["explanation"]
    ben.summary = result["summary"]
    ben.model_version = result["model_version"]

    if not created and ben.statut_revue != "en_attente":
        session.add(Review(beneficiary=ben, reviewer="systeme", action="reinitialise",
                           decision_avant=ben.final_decision, decision_apres=result["eligible"],
                           commentaire="Réponses modifiées dans Kobo : nouvelle évaluation du modèle, revue à refaire."))
    ben.final_decision = result["eligible"]
    ben.statut_revue = "en_attente"
    ben.reviewed_by = None
    ben.reviewed_at = None
    session.commit()
    return ben, created


def apply_review(session: Session, ben: Beneficiary, reviewer: str, approuve: bool, commentaire: str | None) -> Review:
    """Approuver = garder la décision du modèle ; désapprouver = l'inverser (commentaire obligatoire)."""
    commentaire = (commentaire or "").strip() or None
    if not approuve and not commentaire:
        raise ValueError("Un commentaire est obligatoire pour désapprouver la décision du modèle.")
    new_decision = ben.model_decision if approuve else not ben.model_decision
    review = Review(beneficiary=ben, reviewer=reviewer, action="approuve" if approuve else "desapprouve",
                    decision_avant=ben.final_decision, decision_apres=new_decision, commentaire=commentaire)
    session.add(review)
    ben.final_decision = new_decision
    ben.statut_revue = "confirmee" if approuve else "modifiee"
    ben.reviewed_by = reviewer
    ben.reviewed_at = utcnow()
    session.commit()
    return review


def stats(session: Session) -> dict:
    count = lambda *where: session.scalar(select(func.count(Beneficiary.id)).where(*where)) or 0
    total = count()
    revus = count(Beneficiary.statut_revue != "en_attente")
    modifies = count(Beneficiary.statut_revue == "modifiee")
    return {
        "total": total,
        "eligibles_modele": count(Beneficiary.model_decision.is_(True)),
        "eligibles_final": count(Beneficiary.final_decision.is_(True)),
        "en_attente": count(Beneficiary.statut_revue == "en_attente"),
        "a_verifier": count(Beneficiary.a_verifier.is_(True), Beneficiary.statut_revue == "en_attente"),
        "revus": revus,
        "modifies": modifies,
        "taux_desaccord": (modifies / revus) if revus else None,
    }


def serialize(ben: Beneficiary, with_details: bool = True) -> dict:
    data = {
        "id": ben.id,
        "kobo_key": ben.kobo_key,
        "nom_chef_menage": ben.nom_chef_menage,
        "village": ben.village,
        "telephone": ben.telephone,
        "enumerateur": ben.enumerateur,
        "submitted_at": ben.submitted_at.isoformat() if ben.submitted_at else None,
        "score": ben.score,
        "seuil": ben.threshold,
        "decision_modele": "eligible" if ben.model_decision else "non_eligible",
        "decision_finale": "eligible" if ben.final_decision else "non_eligible",
        "statut_revue": ben.statut_revue,
        "a_verifier": ben.a_verifier,
        "reviewed_by": ben.reviewed_by,
        "model_version": ben.model_version,
    }
    if with_details:
        data.update({
            "reponses": ben.features,
            "explication": ben.explanation,
            "resume": ben.summary,
            "valeur_de_base": ben.base_value,
            "revues": [{
                "reviewer": r.reviewer, "action": r.action, "commentaire": r.commentaire,
                "decision_apres": "eligible" if r.decision_apres else "non_eligible",
                "date": r.created_at.isoformat(),
            } for r in ben.reviews],
        })
    return data
