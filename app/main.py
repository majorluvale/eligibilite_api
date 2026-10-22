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
from .db import Beneficiary, Selection, get_session, init_db
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
    indice_priorite: float
    methode_priorite: str


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


# ------------------------------------------------- Sélection sous contrainte budgétaire

class SelectionIn(BaseModel):
    nombre: int
    nom: str = ""
    repartition: str = "globale"  # ou "proportionnelle_zone"
    inclure_non_revus: bool = True


def serialize_selection(sel: Selection, details: bool = True) -> dict:
    data = {"id": sel.id, "nom": sel.nom, "date": sel.created_at.isoformat(), "auteur": sel.created_by,
            "nombre_demande": sel.nb_demande, "eligibles": sel.nb_eligibles, "retenus": sel.nb_retenus,
            "repartition": sel.repartition, "inclure_non_revus": sel.inclure_non_revus, "model_version": sel.model_version,
            "methode_priorite": sel.methode_priorite}
    if details:
        retenus = sorted((i for i in sel.items if i.retenu), key=lambda i: -i.priorite)
        data["menages_retenus"] = [{"rang": n, "rang_dans_la_zone": i.rang if sel.repartition == "proportionnelle_zone" else None,
                                    "id": i.beneficiary_id, "kobo_key": i.beneficiary.kobo_key,
                                    "nom_chef_menage": i.beneficiary.nom_chef_menage, "zone": i.zone,
                                    "indice_priorite": round(i.priorite, 3)} for n, i in enumerate(retenus, 1)]
    return data


@app.post("/api/selections", tags=["sélection"])
def api_create_selection(body: SelectionIn, session: Session = Depends(get_session), user: str = Depends(current_user)):
    try:
        sel = services.creer_selection(session, body.nom, body.nombre, body.repartition, body.inclure_non_revus, user)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return serialize_selection(sel)


@app.get("/api/selections/{sel_id}", tags=["sélection"])
def api_get_selection(sel_id: int, session: Session = Depends(get_session), user: str = Depends(current_user)):
    return serialize_selection(get_selection_or_404(session, sel_id))


def get_selection_or_404(session: Session, sel_id: int) -> Selection:
    sel = session.get(Selection, sel_id)
    if not sel:
        raise HTTPException(404, "Sélection introuvable")
    return sel


@app.get("/dashboard/selections", response_class=HTMLResponse, include_in_schema=False)
def selections_page(request: Request, err: str | None = None, session: Session = Depends(get_session),
                    user: str = Depends(current_user)):
    s = services.stats(session)
    non_revus = session.scalar(select(func.count(Beneficiary.id)).where(
        Beneficiary.final_decision.is_(True), Beneficiary.statut_revue == "en_attente")) or 0
    historique = session.scalars(select(Selection).order_by(Selection.created_at.desc()).limit(50)).all()
    model = load_model(get_settings().model_dir)
    return templates.TemplateResponse(request, "selections.html", {
        "user": user, "csrf": csrf_token(user), "eligibles": s["eligibles_final"], "non_revus": non_revus,
        "historique": historique, "err": err, "methode": model.priority_method,
        "prio": model.metadata.get("priorite", {})})


@app.post("/dashboard/selections", include_in_schema=False)
def selections_create(nombre: int = Form(...), nom: str = Form(""), repartition: str = Form("globale"),
                      inclure_non_revus: str | None = Form(None), csrf: str = Form(""),
                      session: Session = Depends(get_session), user: str = Depends(current_user)):
    check_csrf(user, csrf)
    try:
        sel = services.creer_selection(session, nom, nombre, repartition, inclure_non_revus is not None, user)
    except ValueError as e:
        return RedirectResponse(f"/dashboard/selections?{urlencode({'err': str(e)})}", status_code=303)
    return RedirectResponse(f"/dashboard/selections/{sel.id}", status_code=303)


@app.get("/dashboard/selections/{sel_id}", response_class=HTMLResponse, include_in_schema=False)
def selection_page(request: Request, sel_id: int, session: Session = Depends(get_session),
                   user: str = Depends(current_user)):
    sel = get_selection_or_404(session, sel_id)
    retenus = sorted((i for i in sel.items if i.retenu), key=lambda i: -i.priorite)
    # ménages éligibles juste sous la ligne de coupure : ceux qu'un superviseur doit regarder en premier.
    # Par zone, ce sont les premiers non retenus de chaque zone (rang dans la zone), pas les plus forts indices globaux.
    par_zone = sel.repartition == "proportionnelle_zone"
    non_retenus = sorted((i for i in sel.items if not i.retenu), key=lambda i: -i.priorite)
    if par_zone:  # les 3 premiers non retenus de chaque zone
        sous_la_ligne, vus = [], {}
        for i in non_retenus:
            vus[i.zone] = vus.get(i.zone, 0) + 1
            if vus[i.zone] <= 3:
                sous_la_ligne.append(i)
        sous_la_ligne.sort(key=lambda i: (i.zone, i.rang))
    else:
        sous_la_ligne = non_retenus[:20]
    return templates.TemplateResponse(request, "selection.html", {
        "user": user, "sel": sel, "resume": services.resume_selection(sel), "retenus": retenus,
        "sous_la_ligne": sous_la_ligne, "par_zone": par_zone})


@app.get("/dashboard/selections/{sel_id}/export.csv", include_in_schema=False)
def selection_export(sel_id: int, session: Session = Depends(get_session), user: str = Depends(current_user)):
    sel = get_selection_or_404(session, sel_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rang", "rang_dans_la_zone", "id", "kobo_key", "nom_chef_menage", "telephone", "village", "zone",
                "indice_priorite", "score_eligibilite", "statut_revue", "decision_finale_actuelle", "principaux_facteurs"])
    retenus = sorted((i for i in sel.items if i.retenu), key=lambda i: -i.priorite)
    for rang, i in enumerate(retenus, 1):
        b = i.beneficiary
        facteurs = "; ".join(f"{e['label']} = {e['value']}" for e in b.explanation if e["contribution"] > 0)
        w.writerow([rang, i.rang if sel.repartition == "proportionnelle_zone" else "", b.id, b.kobo_key, b.nom_chef_menage or "", b.telephone or "", b.village or "", i.zone,
                    round(i.priorite, 3), b.score, b.statut_revue, int(b.final_decision), facteurs])
    nom_fichier = f"selection_{sel.id}.csv"
    return StreamingResponse(iter(["﻿" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={nom_fichier}"})


@app.exception_handler(ValidationError)
def validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
