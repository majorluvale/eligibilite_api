import copy

import pytest

from app.security import csrf_token
from app.services import repartir
from tests.test_api import KOBO_PAYLOAD, auth, client  # noqa: F401  (fixture partagée)

# Ménages éligibles de vulnérabilité croissante (score de la grille : environ 42, 48, 55 et 78 points)
_BASE = {"profil/statut_menage": "resident", "profil/sexe_chef_menage": "femme", "profil/taille_menage": "8",
         "profil/enfants_moins_5": "3", "vulnerabilites/femme_enceinte_allaitante": "non",
         "logement/type_logement": "maison_endommagee", "logement/articles_essentiels": "3",
         "alimentation_revenus/repas_par_jour": "2", "alimentation_revenus/jours_sans_nourriture_7j": "1"}
PROFILS = [
    {**_BASE, "profil/localite": "Kibirizi"},
    {**_BASE, "profil/localite": "Kibirizi", "logement/type_logement": "abri_fortune"},
    {**_BASE, "profil/localite": "Binza", "logement/type_logement": "abri_fortune", "alimentation_revenus/repas_par_jour": "1"},
    {**_BASE, "profil/localite": "Binza", "logement/type_logement": "abri_fortune", "alimentation_revenus/repas_par_jour": "1",
     "profil/statut_menage": "deplace", "profil/duree_deplacement_mois": "1", "vulnerabilites/membre_handicap": "oui"},
]


@pytest.fixture(scope="module")
def menages(client):
    ids = []
    for i, profil in enumerate(PROFILS):
        p = {k: v for k, v in copy.deepcopy(KOBO_PAYLOAD).items() if k != "profil/duree_deplacement_mois"}
        p.update(profil)
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
