import base64
import copy
import os
import tempfile
from pathlib import Path

import openpyxl
import pytest

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp) / 'test.db'}"
os.environ["DASHBOARD_USERS"] = "agent:secret,superviseur:pw2"
os.environ["KOBO_WEBHOOK_TOKEN"] = "tok123"
os.environ["SECRET_KEY"] = "test-key"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schema import FEATURES  # noqa: E402
from app.security import csrf_token  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Soumission telle qu'envoyée par un REST Service KoboToolbox (groupes préfixés, valeurs en texte)
KOBO_PAYLOAD = {
    "_id": 4242,
    "_uuid": "b1f0c6a2-0000-4c3b-9d1e-123456789abc",
    "_submission_time": "2026-10-20T09:15:02",
    "_xform_id_string": "ciblage_menages_eligibilite",
    "_geolocation": [-1.2, 29.4],
    "_attachments": [],
    "formhub/uuid": "abc",
    "start": "2026-10-20T10:00:00.000+02:00",
    "end": "2026-10-20T10:12:00.000+02:00",
    "consentement": "oui",
    "identification/enumerateur": "Grace",
    "identification/village": "Kibirizi",
    "identification/nom_chef_menage": "Amani Kahindo",
    "identification/telephone": "+243 990 000 000",
    "identification/gps": "-1.2 29.4 1500 5",
    "menage/location": "zone_f",
    "menage/statut_menage": "idp",
    "menage/sexe_chef_menage": "f",
    "menage/age_chef_menage": "16",
    "menage/taille_menage": "9",
    "menage/membre_handicap": "non",
    "menage/femme_enceinte_allaitante": "oui",
    "conditions_vie/type_logement": "urgence",
    "conditions_vie/source_eau": "surface",
    "economie/source_revenu": "journalier",
    "economie/nombre_repas_jour": "1",
    "economie/dette_contractee_crise": "oui",
    "meta/instanceID": "uuid:b1f0c6a2-0000-4c3b-9d1e-123456789abc",
}

NON_VULNERABLE = {
    "location": "zone_a", "statut_menage": "hote", "sexe_chef_menage": "m", "age_chef_menage": 40,
    "taille_menage": 4, "membre_handicap": "non", "femme_enceinte_allaitante": "non",
    "type_logement": "dur", "source_eau": "robinet", "source_revenu": "salarie",
    "nombre_repas_jour": 3, "dette_contractee_crise": "non",
}


def auth(user="agent", pw="secret"):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_webhook_requires_token(client):
    assert client.post("/api/kobo/webhook", json=KOBO_PAYLOAD).status_code == 401
    assert client.post("/api/kobo/webhook?token=faux", json=KOBO_PAYLOAD).status_code == 401


def test_webhook_scores_and_is_idempotent(client):
    r = client.post("/api/kobo/webhook?token=tok123", json=KOBO_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cree"] is True and body["eligible"] is True
    again = client.post("/api/kobo/webhook", json=KOBO_PAYLOAD, headers={"X-Webhook-Token": "tok123"})
    assert again.json() == {**body, "cree": False}
    assert len(client.get("/api/beneficiaires", headers=auth()).json()) == 1


def test_webhook_rejects_invalid_answers(client):
    bad = {**KOBO_PAYLOAD, "_id": 1, "menage/location": "zone_z", "menage/age_chef_menage": "abc"}
    r = client.post("/api/kobo/webhook?token=tok123", json=bad)
    assert r.status_code == 422
    assert "location" in r.json()["detail"] and "age_chef_menage" in r.json()["detail"]


def test_webhook_ignores_refused_consent(client):
    r = client.post("/api/kobo/webhook?token=tok123", json={"_id": 9, "consentement": "non"})
    assert r.status_code == 200 and r.json()["ignore"] is True


def test_predict_explanation_is_additive(client):
    import math
    r = client.post("/api/predict", json=NON_VULNERABLE).json()
    assert r["eligible"] is False
    logit = r["base_value"] + sum(e["contribution"] for e in r["explanation"])
    assert abs(1 / (1 + math.exp(-logit)) - r["score"]) < 1e-3
    assert {e["feature"] for e in r["explanation"]} == {f.name for f in FEATURES}


def test_human_override_and_audit_trail(client):
    ben_id = client.get("/api/beneficiaires", headers=auth()).json()[0]["id"]
    # désapprouver sans commentaire est refusé
    r = client.post(f"/api/beneficiaires/{ben_id}/revision", json={"approuve": False}, headers=auth())
    assert r.status_code == 422
    r = client.post(f"/api/beneficiaires/{ben_id}/revision", headers=auth("superviseur", "pw2"),
                    json={"approuve": False, "commentaire": "Ménage déjà assisté par un autre partenaire"})
    d = r.json()
    assert d["decision_modele"] == "eligible" and d["decision_finale"] == "non_eligible"
    assert d["statut_revue"] == "modifiee" and d["reviewed_by"] == "superviseur"
    assert d["revues"][0]["commentaire"].startswith("Ménage déjà")


def test_kobo_edit_resets_review(client):
    edited = copy.deepcopy(KOBO_PAYLOAD)
    edited["economie/nombre_repas_jour"] = "3"
    client.post("/api/kobo/webhook?token=tok123", json=edited)
    d = client.get("/api/beneficiaires", headers=auth()).json()[0]
    assert d["statut_revue"] == "en_attente"


def test_dashboard_pages_and_csrf(client):
    assert client.get("/dashboard").status_code == 401
    assert client.get("/dashboard", headers=auth("agent", "mauvais")).status_code == 401
    page = client.get("/dashboard", headers=auth())
    assert page.status_code == 200 and "Amani Kahindo" in page.text
    ben_id = client.get("/api/beneficiaires", headers=auth()).json()[0]["id"]
    detail = client.get(f"/dashboard/beneficiaires/{ben_id}", headers=auth())
    assert detail.status_code == 200 and "Pourquoi le modèle" in detail.text
    assert client.get("/dashboard/modele", headers=auth()).status_code == 200
    # formulaire sans jeton CSRF refusé, avec jeton accepté
    url = f"/dashboard/beneficiaires/{ben_id}/revision"
    assert client.post(url, data={"decision": "approuve"}, headers=auth()).status_code == 403
    r = client.post(url, data={"decision": "approuve", "csrf": csrf_token("agent")}, headers=auth(),
                    follow_redirects=False)
    assert r.status_code == 303
    csv_text = client.get("/dashboard/export.csv", headers=auth()).text
    assert "Amani Kahindo" in csv_text and "confirmee" in csv_text


def test_xlsform_matches_schema():
    from kobo.build_xlsform import build
    wb = openpyxl.load_workbook(build())
    survey = list(wb["survey"].iter_rows(values_only=True))
    header = survey[0]
    names = {r[header.index("name")]: r[header.index("type")] for r in survey[1:]}
    choices = {}
    for list_name, name, _ in list(wb["choices"].iter_rows(values_only=True))[1:]:
        choices.setdefault(list_name, set()).add(name)
    for f in FEATURES:
        assert f.kobo in names, f"question {f.kobo} absente du formulaire"
        qtype = names[f.kobo]
        if f.kind == "categorical":
            assert choices[qtype.split()[1]] == set(f.choices), f.kobo
