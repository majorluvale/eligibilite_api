"""Logique métier : évaluer une soumission, enregistrer une revue humaine, statistiques."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import kobo
from .config import get_settings
from .db import Beneficiary, Review, Selection, SelectionItem, utcnow
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


def priorites(menages: list[Beneficiary]) -> tuple[list[float], str]:
    """Indice de priorité de chaque ménage, calculé en un seul passage par le modèle.

    Méthode principale : le score de l'agent (points de sa grille Excel) prédit par le modèle de
    priorité. On ne classe pas par probabilité d'éligibilité : elle mesure la certitude du modèle,
    pas l'intensité des besoins, et vaut presque 1 pour beaucoup de ménages.
    """
    model = load_model(get_settings().model_dir)
    valeurs = model.priorities(pd.DataFrame([b.features for b in menages]))
    return [float(v) for v in valeurs], model.priority_method


def repartir(n: int, effectifs: dict[str, int]) -> dict[str, int]:
    """Quotas proportionnels au nombre d'éligibles par zone (méthode du plus fort reste)."""
    total = sum(effectifs.values())
    if n >= total:
        return dict(effectifs)
    exacts = {z: n * c / total for z, c in effectifs.items()}
    quotas = {z: int(q) for z, q in exacts.items()}
    reste = n - sum(quotas.values())
    for z in sorted(exacts, key=lambda z: exacts[z] - quotas[z], reverse=True)[:reste]:
        quotas[z] += 1
    return quotas


def creer_selection(session: Session, nom: str, nb: int, repartition: str, inclure_non_revus: bool,
                    auteur: str) -> Selection:
    """Classe les ménages éligibles par indice de priorité et retient les `nb` premiers.

    - repartition="globale" : les `nb` ménages les plus prioritaires, toutes zones confondues ;
    - repartition="proportionnelle_zone" : chaque zone reçoit un quota proportionnel à son
      nombre d'éligibles, puis on prend les plus prioritaires dans chaque zone.
    Ex aequo : le ménage enquêté le plus tôt passe d'abord.
    """
    if nb < 1:
        raise ValueError("Le nombre de ménages à sélectionner doit être au moins 1.")
    if repartition not in ("globale", "proportionnelle_zone"):
        raise ValueError("Répartition inconnue.")
    query = select(Beneficiary).where(Beneficiary.final_decision.is_(True))
    if not inclure_non_revus:
        query = query.where(Beneficiary.statut_revue != "en_attente")
    eligibles = session.scalars(query).all()
    if not eligibles:
        raise ValueError("Aucun ménage éligible ne correspond à ces critères.")

    valeurs, methode = priorites(eligibles)
    prio = {b.id: v for b, v in zip(eligibles, valeurs)}
    ordre = sorted(eligibles, key=lambda b: (-prio[b.id], b.submitted_at or b.received_at, b.id))

    items = []
    if repartition == "globale":
        for rang, b in enumerate(ordre, 1):
            items.append(SelectionItem(beneficiary=b, rang=rang, priorite=prio[b.id],
                                       zone=b.features["Location"], retenu=rang <= nb))
    else:
        par_zone: dict[str, list[Beneficiary]] = {}
        for b in ordre:
            par_zone.setdefault(b.features["Location"], []).append(b)
        quotas = repartir(nb, {z: len(m) for z, m in par_zone.items()})
        for zone, menages in par_zone.items():
            for rang, b in enumerate(menages, 1):
                items.append(SelectionItem(beneficiary=b, rang=rang, priorite=prio[b.id], zone=zone,
                                           retenu=rang <= quotas[zone]))

    sel = Selection(nom=nom.strip() or f"Sélection de {nb} ménages", created_by=auteur, nb_demande=nb,
                    nb_eligibles=len(eligibles), nb_retenus=sum(i.retenu for i in items), repartition=repartition,
                    inclure_non_revus=inclure_non_revus, model_version=load_model(get_settings().model_dir).version,
                    methode_priorite=methode, items=items)
    session.add(sel)
    session.commit()
    return sel


def resume_selection(sel: Selection) -> dict:
    """Couverture par zone et seuil de priorité de la dernière place retenue."""
    zones: dict[str, dict] = {}
    for it in sel.items:
        z = zones.setdefault(it.zone, {"eligibles": 0, "retenus": 0, "seuil": None})
        z["eligibles"] += 1
        if it.retenu:
            z["retenus"] += 1
            z["seuil"] = it.priorite if z["seuil"] is None else min(z["seuil"], it.priorite)
    retenus = [i.priorite for i in sel.items if i.retenu]
    return {"zones": dict(sorted(zones.items())), "seuil_global": min(retenus) if retenus else None}


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
