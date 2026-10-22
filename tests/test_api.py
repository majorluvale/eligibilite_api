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
    "consentement": "oui",
    "identification/enumerateur": "Grace",
    "identification/village": "Kibirizi centre",
    "identification/nom_chef_menage": "Amani Kahindo",
    "identification/telephone": "+243 990 000 000",
    "identification/gps": "-1.2 29.4 1500 5",
    "profil/localite": "Kibirizi",
    "profil/statut_menage": "deplace",
    "profil/duree_deplacement_mois": "1",
    "profil/sexe_chef_menage": "femme",
    "profil/age_chef_menage": "34",
    "profil/taille_menage": "8",
    "profil/enfants_moins_5": "3",
    "vulnerabilites/membre_handicap": "non",
    "vulnerabilites/membre_maladie_chronique": "non",
    "vulnerabilites/femme_enceinte_allaitante": "oui",
    "vulnerabilites/enfant_non_accompagne_heberge": "non",
    "logement/type_logement": "abri_fortune",
    "logement/articles_essentiels": "1",
    "logement/possede_betail": "non",
    "alimentation_revenus/repas_par_jour": "1",
    "alimentation_revenus/jours_sans_nourriture_7j": "3",
    "alimentation_revenus/source_revenu": "travail_journalier",
    "assistance/assistance_3_derniers_mois": "non",
    "meta/instanceID": "uuid:b1f0c6a2-0000-4c3b-9d1e-123456789abc",
}

NON_VULNERABLE = {
    "localite": "Rutshuru", "statut_menage": "resident", "sexe_chef_menage": "homme", "age_chef_menage": 40,
    "taille_menage": 3, "enfants_moins_5": 0, "membre_handicap": "non", "membre_maladie_chronique": "non",
    "femme_enceinte_allaitante": "non", "enfant_non_accompagne_heberge": "non", "type_logement": "maison_correcte",
    "articles_essentiels": 5, "repas_par_jour": 3, "jours_sans_nourriture_7j": 0, "source_revenu": "petit_commerce",
    "possede_betail": "oui", "assistance_3_derniers_mois": "non",
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
    bad = {**KOBO_PAYLOAD, "_id": 1, "profil/localite": "Goma", "profil/age_chef_menage": "abc"}
    r = client.post("/api/kobo/webhook?token=tok123", json=bad)
    assert r.status_code == 422
    assert "localite" in r.json()["detail"] and "age_chef_menage" in r.json()["detail"]


def test_questions_conditionnelles_et_coherence(client):
    # déplacé sans durée de déplacement : refusé
    sans_duree = {k: v for k, v in KOBO_PAYLOAD.items() if k != "profil/duree_deplacement_mois"}
    assert "duree_deplacement_mois" in client.post("/api/predict", json=sans_duree).json()["detail"]
    # résident : la durée n'est pas demandée ; assistance reçue : le type devient obligatoire
    resident = {**NON_VULNERABLE, "assistance_3_derniers_mois": "oui"}
    assert "type_assistance_recue" in client.post("/api/predict", json=resident).json()["detail"]
    assert client.post("/api/predict", json={**resident, "type_assistance_recue": "abri"}).status_code == 200
    # plus d'enfants de moins de 5 ans que de membres : refusé
    assert "enfants_moins_5" in client.post("/api/predict", json={**NON_VULNERABLE, "enfants_moins_5": 3}).json()["detail"]


def test_webhook_ignores_refused_consent(client):
    r = client.post("/api/kobo/webhook?token=tok123", json={"_id": 9, "consentement": "non"})
    assert r.status_code == 200 and r.json()["ignore"] is True


def test_predict_explication_en_points(client):
    r = client.post("/api/predict", json=NON_VULNERABLE).json()
    assert r["eligible"] is False and r["points"] < r["seuil_grille"]
    assert abs(r["base_value"] + sum(e["contribution"] for e in r["explanation"]) - r["points"]) < 0.1
    assert {e["feature"] for e in r["explanation"]} == {f.name for f in FEATURES if f.modele}
    assert r["indice_priorite"] == pytest.approx(r["points"], abs=0.05)


def test_exception_de_la_grille(client):
    # ménage très vulnérable mais avec un salaire régulier : exclu par la grille de l'ONG
    salarie = {k.split("/")[-1]: v for k, v in KOBO_PAYLOAD.items() if "/" in k}
    salarie["source_revenu"] = "salaire_regulier"
    r = client.post("/api/predict", json=salarie).json()
    assert r["eligible"] is False and r["points"] >= r["seuil_grille"]
    assert r["exception"]["type"] == "exclusion" and "Source de revenu" in r["exception"]["facteur"]


def test_human_override_and_audit_trail(client):
    ben_id = client.get("/api/beneficiaires", headers=auth()).json()[0]["id"]
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
    edited["alimentation_revenus/repas_par_jour"] = "3"
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
    assert detail.status_code == 200 and "Pourquoi le modèle" in detail.text and "Abri de fortune" in detail.text
    assert client.get("/dashboard/modele", headers=auth()).status_code == 200
    url = f"/dashboard/beneficiaires/{ben_id}/revision"
    assert client.post(url, data={"decision": "approuve"}, headers=auth()).status_code == 403
    r = client.post(url, data={"decision": "approuve", "csrf": csrf_token("agent")}, headers=auth(),
                    follow_redirects=False)
    assert r.status_code == 303
    csv_text = client.get("/dashboard/export.csv", headers=auth()).text
    assert "Amani Kahindo" in csv_text and "confirmee" in csv_text


def test_xlsform_matches_schema(tmp_path, monkeypatch):
    from kobo import build_xlsform
    monkeypatch.setattr(build_xlsform, "OUT", tmp_path / "formulaire.xlsx")  # ne touche pas au fichier versionné
    wb = openpyxl.load_workbook(build_xlsform.build())
    survey = list(wb["survey"].iter_rows(values_only=True))
    header = survey[0]
    rows = {r[header.index("name")]: r for r in survey[1:]}
    choices = {}
    for list_name, name, _ in list(wb["choices"].iter_rows(values_only=True))[1:]:
        choices.setdefault(list_name, set()).add(name)
    for f in FEATURES:
        assert f.name in rows, f"question {f.name} absente du formulaire"
        qtype = rows[f.name][header.index("type")]
        if f.kind == "categorical":
            assert choices[qtype.split()[1]] == set(f.choices), f.name
        if f.relevant:
            assert f"${{{f.relevant[0]}}}" in rows[f.name][header.index("relevant")], f.name
