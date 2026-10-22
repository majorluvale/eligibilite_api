"""API FastAPI : webhook Kobo, prédiction, revue humaine et dashboard d'explicabilité."""
from __future__ import annotations

import csv
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from . import kobo, services
from .config import get_settings
from .db import Beneficiary, get_session, init_db
from .ml import load_metadata, load_model
from .schema import FEATURES, IDENTITY_FIELDS, SHORT_LABELS, ValidationError, normalize_features
from .security import check_csrf, check_webhook_token, csrf_token, current_user

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("eligibilite")
HERE = Path(__file__).resolve().parent
PAGE_SIZE = 25


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model(get_settings().model_dir)  # charge le modèle au démarrage, pas à la 1re requête
    yield


app = FastAPI(
    title="API d'éligibilité des ménages",
    description="Reçoit les soumissions KoboToolbox, prédit l'éligibilité, explique la décision "
                "et permet à un humain de la confirmer ou de la désapprouver.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")
templates.env.globals["SHORT_LABELS"] = SHORT_LABELS


def fmt_pct(v):
    return "—" if v is None else f"{v:.1%}".replace(".", ",")


templates.env.filters["pct"] = fmt_pct


# ----------------------------------------------------------------------------- API

@app.get("/health", tags=["système"])
def health():
    return {"status": "ok", "model_version": load_model(get_settings().model_dir).version}


@app.post("/api/kobo/webhook", tags=["kobo"], dependencies=[Depends(check_webhook_token)])
def kobo_webhook(payload: dict = Body(...), session: Session = Depends(get_session)):
    """Point d'entrée du REST Service KoboToolbox (format JSON)."""
    if kobo.consent_refused(payload):
        return {"ignore": True, "raison": "consentement refusé"}
    try:
        ben, created = services.ingest_submission(session, payload, source="kobo_webhook")
    except (ValidationError, ValueError) as e:
        log.warning("Soumission Kobo rejetée : %s", e)
        raise HTTPException(422, str(e))
    return {"id": ben.id, "cree": created, "eligible": ben.final_decision, "score": ben.score,
            "a_verifier": ben.a_verifier}


class PredictionOut(BaseModel):
    eligible: bool
    score: float
    threshold: float
    a_verifier: bool
    base_value: float
    explanation: list[dict]
    summary: str
    model_version: str


@app.post("/api/predict", tags=["modèle"], response_model=PredictionOut)
def predict(reponses: dict = Body(..., examples=[{
    "location": "zone_e", "statut_menage": "idp", "sexe_chef_menage": "f", "age_chef_menage": 34,
    "taille_menage": 9, "membre_handicap": "non", "femme_enceinte_allaitante": "oui",
    "type_logement": "transitionnel", "source_eau": "puits_ouvert", "source_revenu": "journalier",
    "nombre_repas_jour": 1, "dette_contractee_crise": "oui"}])):
    """Prédiction sans enregistrement. Accepte les codes Kobo ou les valeurs d'entraînement."""
    try:
        features = normalize_features(kobo.flatten(reponses))
    except ValidationError as e:
        raise HTTPException(422, str(e))
    s = get_settings()
    return load_model(s.model_dir).predict(features, review_band=s.review_band)


@app.get("/api/modele", tags=["modèle"])
def model_info():
    return load_metadata(get_settings().model_dir)


@app.get("/api/beneficiaires", tags=["bénéficiaires"])
def api_list(statut: str | None = None, decision: str | None = None, limit: int = 100, offset: int = 0,
             session: Session = Depends(get_session), user: str = Depends(current_user)):
    q = filtered_query(statut=statut, decision=decision)
    rows = session.scalars(q.order_by(Beneficiary.received_at.desc()).limit(min(limit, 1000)).offset(offset))
    return [services.serialize(b, with_details=False) for b in rows]


@app.get("/api/beneficiaires/{ben_id}", tags=["bénéficiaires"])
def api_detail(ben_id: int, session: Session = Depends(get_session), user: str = Depends(current_user)):
    return services.serialize(get_or_404(session, ben_id))


class ReviewIn(BaseModel):
    approuve: bool
    commentaire: str | None = None


@app.post("/api/beneficiaires/{ben_id}/revision", tags=["bénéficiaires"])
def api_review(ben_id: int, body: ReviewIn, session: Session = Depends(get_session),
               user: str = Depends(current_user)):
    ben = get_or_404(session, ben_id)
    try:
        services.apply_review(session, ben, user, body.approuve, body.commentaire)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return services.serialize(ben)


@app.post("/api/kobo/sync", tags=["kobo"])
def api_sync(session: Session = Depends(get_session), user: str = Depends(current_user)):
    return run_sync(session)


def run_sync(session: Session) -> dict:
    try:
        submissions = kobo.fetch_submissions()
    except Exception as e:  # réseau, jeton invalide, configuration manquante
        raise HTTPException(502, f"Synchronisation Kobo impossible : {e}")
    created = updated = rejected = 0
    for payload in submissions:
        if kobo.consent_refused(payload):
            continue
        try:
            _, was_created = services.ingest_submission(session, payload, source="kobo_sync")
            created += was_created
            updated += not was_created
        except (ValidationError, ValueError) as e:
            session.rollback()
            rejected += 1
            log.warning("Soumission %s rejetée : %s", payload.get("_id"), e)
    return {"recues": len(submissions), "nouvelles": created, "deja_presentes": updated, "rejetees": rejected}


# ----------------------------------------------------------------------- Dashboard

def get_or_404(session: Session, ben_id: int) -> Beneficiary:
    ben = session.get(Beneficiary, ben_id)
    if not ben:
        raise HTTPException(404, "Ménage introuvable")
    return ben


def filtered_query(q: str | None = None, statut: str | None = None, decision: str | None = None,
                   a_verifier: bool = False):
    query = select(Beneficiary)
    if q:
        like = f"%{q.strip()}%"
        query = query.where(or_(Beneficiary.nom_chef_menage.ilike(like), Beneficiary.village.ilike(like),
                                Beneficiary.kobo_key.ilike(like), Beneficiary.telephone.ilike(like)))
    if statut:
        query = query.where(Beneficiary.statut_revue == statut)
    if decision in ("eligible", "non_eligible"):
        query = query.where(Beneficiary.final_decision.is_(decision == "eligible"))
    if a_verifier:
        query = query.where(Beneficiary.a_verifier.is_(True))
    return query


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/dashboard")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard(request: Request, q: str | None = None, statut: str | None = None, decision: str | None = None,
              a_verifier: bool = False, page: int = 1, msg: str | None = None,
              session: Session = Depends(get_session), user: str = Depends(current_user)):
    query = filtered_query(q, statut, decision, a_verifier)
    total = session.scalar(select(func.count()).select_from(query.subquery()))
    page = max(page, 1)
    rows = session.scalars(query.order_by(Beneficiary.a_verifier.desc(), Beneficiary.received_at.desc())
                           .limit(PAGE_SIZE).offset((page - 1) * PAGE_SIZE)).all()
    params = {k: v for k, v in {"q": q, "statut": statut, "decision": decision,
                                 "a_verifier": "true" if a_verifier else None}.items() if v}
    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user, "stats": services.stats(session), "rows": rows, "total": total, "page": page,
        "pages": max(1, -(-total // PAGE_SIZE)), "filters": params, "qs": urlencode(params),
        "msg": msg, "csrf": csrf_token(user), "sync_enabled": bool(get_settings().kobo_api_token),
    })


@app.get("/dashboard/beneficiaires/{ben_id}", response_class=HTMLResponse, include_in_schema=False)
def beneficiary_page(request: Request, ben_id: int, msg: str | None = None, err: str | None = None,
                     session: Session = Depends(get_session), user: str = Depends(current_user)):
    ben = get_or_404(session, ben_id)
    max_abs = max((abs(e["contribution"]) for e in ben.explanation), default=1.0) or 1.0
    return templates.TemplateResponse(request, "beneficiaire.html", {
        "user": user, "b": ben, "features": FEATURES, "identity_fields": IDENTITY_FIELDS,
        "max_abs": max_abs, "csrf": csrf_token(user), "msg": msg, "err": err,
    })


@app.post("/dashboard/beneficiaires/{ben_id}/revision", include_in_schema=False)
def beneficiary_review(ben_id: int, decision: str = Form(...), commentaire: str = Form(""),
                       csrf: str = Form(""), session: Session = Depends(get_session),
                       user: str = Depends(current_user)):
    check_csrf(user, csrf)
    ben = get_or_404(session, ben_id)
    try:
        services.apply_review(session, ben, user, decision == "approuve", commentaire)
    except ValueError as e:
        return RedirectResponse(f"/dashboard/beneficiaires/{ben_id}?{urlencode({'err': str(e)})}", status_code=303)
    return RedirectResponse(f"/dashboard/beneficiaires/{ben_id}?msg=Revue+enregistr%C3%A9e", status_code=303)


@app.post("/dashboard/sync", include_in_schema=False)
def dashboard_sync(csrf: str = Form(""), session: Session = Depends(get_session), user: str = Depends(current_user)):
    check_csrf(user, csrf)
    try:
        r = run_sync(session)
        msg = (f"Synchronisation : {r['recues']} soumissions, {r['nouvelles']} nouvelles, "
               f"{r['rejetees']} rejetées.")
    except HTTPException as e:
        msg = e.detail
    return RedirectResponse(f"/dashboard?{urlencode({'msg': msg})}", status_code=303)


@app.get("/dashboard/modele", response_class=HTMLResponse, include_in_schema=False)
def model_page(request: Request, user: str = Depends(current_user)):
    meta = load_metadata(get_settings().model_dir)
    max_imp = max((r["importance"] for r in meta["importance_globale"]), default=1.0) or 1.0
    return templates.TemplateResponse(request, "modele.html", {"user": user, "m": meta, "max_imp": max_imp})


@app.get("/dashboard/export.csv", include_in_schema=False)
def export_csv(session: Session = Depends(get_session), user: str = Depends(current_user)):
    buf = io.StringIO()
    feature_cols = [f.name for f in FEATURES]
    w = csv.writer(buf)
    w.writerow(["id", "kobo_key", *IDENTITY_FIELDS, *feature_cols, "score", "seuil", "decision_modele",
                "decision_finale", "statut_revue", "revu_par", "commentaire_revue", "model_version"])
    for b in session.scalars(select(Beneficiary).order_by(Beneficiary.id)):
        last = b.reviews[0].commentaire if b.reviews else ""
        w.writerow([b.id, b.kobo_key, *(getattr(b, k) or "" for k in IDENTITY_FIELDS),
                    *(b.features.get(c) for c in feature_cols), b.score, b.threshold,
                    int(b.model_decision), int(b.final_decision), b.statut_revue, b.reviewed_by or "",
                    last or "", b.model_version])
    return StreamingResponse(iter(["﻿" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=beneficiaires.csv"})


@app.exception_handler(ValidationError)
def validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
