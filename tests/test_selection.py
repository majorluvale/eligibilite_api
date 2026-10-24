import copy

import pytest

from app.main import EXEMPLE_MENAGE
from app.security import csrf_token
from app.services import repartir
from tests.test_api import KOBO_PAYLOAD, auth, client, kobo  # noqa: F401  (fixture partagée)

# Ménages éligibles de vulnérabilité croissante (variantes de l'exemple de l'API)
_MOINS = {"statut_menage": "resident", "type_logement": "maison_endommagee", "statut_occupation": "proprietaire",
          "incident_securite_3_mois": "non", "biens_perdus_crise": "non", "documents_identite": "oui",
          "restriction_mouvements": "non"}
_RESIDENT = {k: v for k, v in EXEMPLE_MENAGE.items() if k not in ("duree_deplacement_mois", "nombre_deplacements_12m")}
PROFILS = [
    {**_RESIDENT, **_MOINS, "localite": "Kibirizi", "rcsi_aliments_moins_chers": 3, "rcsi_reduction_portions": 2,
     "rcsi_moins_de_repas": 2, "rcsi_adultes_pour_enfants": 1, "jours_legumineuses": 3, "jours_huile": 4,
     "revenu_30j_usd": 45, "part_depenses_alimentation_pct": 70, "stock_nourriture_jours": 3},
    {**_RESIDENT, **_MOINS, "localite": "Kibirizi", "rcsi_reduction_portions": 4, "revenu_30j_usd": 35, "jours_legumineuses": 2},
    {**EXEMPLE_MENAGE, "localite": "Binza", "rcsi_reduction_portions": 4},
    {**EXEMPLE_MENAGE, "localite": "Binza", "jours_sans_nourriture_7j": 4, "enfant_malnutrition_depiste": "oui",
     "lcs_mendicite": "oui"},
]
IDENTITE = {k: v for k, v in KOBO_PAYLOAD.items() if k.startswith(("_", "identification/", "meta/", "consentement"))}


@pytest.fixture(scope="module")
def menages(client):
    ids = []
    for i, profil in enumerate(PROFILS):
        p = {**copy.deepcopy(IDENTITE), **kobo(profil)}
        p.update({"_id": 9000 + i, "_uuid": f"sel-{i}", "identification/nom_chef_menage": f"Profil {i}"})
        r = client.post("/api/kobo/webhook?token=tok123", json=p)
        assert r.status_code == 200 and r.json()["eligible"], r.text
        ids.append(r.json()["id"])
    return ids


def test_repartir_plus_fort_reste():
    assert repartir(10, {"A": 50, "B": 30, "C": 20}) == {"A": 5, "B": 3, "C": 2}
    q = repartir(7, {"A": 10, "B": 10, "C": 10})
    assert sum(q.values()) == 7 and max(q.values()) - min(q.values()) <= 1
    assert repartir(100, {"A": 3, "B": 2}) == {"A": 3, "B": 2}  # budget suffisant : tout le monde


def test_priorite_classe_les_plus_vulnerables_en_premier(client, menages):
    r = client.post("/api/selections", headers=auth(), json={"nombre": 2, "nom": "Test global"})
    assert r.status_code == 200, r.text
    sel = r.json()
    assert sel["retenus"] == 2 and sel["eligibles"] >= len(PROFILS)
    retenus = [m["id"] for m in sel["menages_retenus"]]
    assert menages[3] in retenus and menages[0] not in retenus
    indices = [m["indice_priorite"] for m in sel["menages_retenus"]]
    assert indices == sorted(indices, reverse=True)


def test_repartition_par_zone(client, menages):
    sel = client.post("/api/selections", headers=auth(),
                      json={"nombre": 2, "repartition": "proportionnelle_zone"}).json()
    zones = {m["zone"] for m in sel["menages_retenus"]}
    assert len(zones) == 2  # les deux zones présentes reçoivent chacune une place


def test_selection_invalide(client, menages):
    assert client.post("/api/selections", headers=auth(), json={"nombre": 0}).status_code == 422
    assert client.post("/api/selections", headers=auth(), json={"nombre": 5, "repartition": "x"}).status_code == 422
    assert client.post("/api/selections", json={"nombre": 5}).status_code == 401
    assert client.post("/api/selections", headers=auth(), json={"nombre": 5, "assistance": "x"}).status_code == 422


def test_selection_par_type_assistance(client, menages):
    detail = client.get(f"/api/beneficiaires/{menages[3]}", headers=auth()).json()
    code = detail["assistance_ideale"]
    sel = client.post("/api/selections", headers=auth(), json={"nombre": 10, "assistance": code}).json()
    assert sel["assistance"] == code and sel["retenus"] >= 1
    assert {m["assistance_ideale"] for m in sel["menages_retenus"]} == {code}
    assert menages[3] in [m["id"] for m in sel["menages_retenus"]]


def test_pages_et_export(client, menages):
    assert client.get("/dashboard/selections", headers=auth()).status_code == 200
    url = "/dashboard/selections"
    assert client.post(url, data={"nombre": "3"}, headers=auth()).status_code == 403  # sans CSRF
    r = client.post(url, data={"nombre": "3", "nom": "Distribution test", "repartition": "globale",
                               "inclure_non_revus": "1", "csrf": csrf_token("agent")},
                    headers=auth(), follow_redirects=False)
    assert r.status_code == 303
    page = client.get(r.headers["location"], headers=auth())
    assert page.status_code == 200 and "Distribution test" in page.text and "Juste sous la ligne" in page.text
    csv_text = client.get(r.headers["location"] + "/export.csv", headers=auth()).text
    assert "Profil 3" in csv_text and csv_text.count("\n") == 4  # en-tête + 3 ménages
