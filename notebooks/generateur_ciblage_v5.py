"""Générateur de données v5 — ciblage opérationnel d'une ONG (grille de points), sans bruit.

Processus simulé :
1. L'ONG définit ses critères et leurs points, puis les fait valider localement
   (comité de ciblage, autorités, leaders communautaires).
2. Les enquêteurs collectent les données avec Kobo.
3. Un agent applique la grille dans Excel : il additionne les points, applique
   les inclusions automatiques et les exclusions, puis décide « éligible » ou non.
4. Si le budget ne couvre pas tous les éligibles, il classe les éligibles par score.

Les colonnes `score_excel`, `motif_decision` et `eligible` sont ce que l'agent
produit dans Excel. La décision est une fonction exacte des réponses : aucune
variation aléatoire n'est ajoutée.
La grille est un exemple réaliste, pas celle d'une ONG en particulier.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

N_MENAGES = 100_000
GRAINE = 2026
SEUIL_ELIGIBILITE = 40

LOCALITES = {"Rutshuru": 0.35, "Rwanguba": 0.50, "Binza": 0.65, "Bambo": 0.75, "Kibirizi": 0.85, "Birambizo": 0.95}

# (critère, réponse, points) : la grille telle qu'elle figure dans la feuille Excel de l'agent
GRILLE = [
    ("statut_menage", "deplace depuis moins de 3 mois", 15), ("statut_menage", "deplace depuis 3 mois ou plus", 10),
    ("statut_menage", "retourne", 8), ("statut_menage", "famille d'accueil", 5), ("statut_menage", "resident", 0),
    ("chef_menage", "femme", 6), ("chef_menage", "mineur (< 18 ans)", 12), ("chef_menage", "age (>= 60 ans)", 8),
    ("taille_menage", "1 a 3 personnes", 0), ("taille_menage", "4 a 6", 4), ("taille_menage", "7 a 9", 7), ("taille_menage", "10 et plus", 10),
    ("enfants_moins_5", "par enfant (max 6 points)", 2),
    ("membre_handicap", "oui", 8), ("membre_maladie_chronique", "oui", 5), ("femme_enceinte_allaitante", "oui", 5),
    ("enfant_non_accompagne_heberge", "oui", 8),
    ("type_logement", "sans abri", 15), ("type_logement", "abri de fortune", 10), ("type_logement", "heberge par une famille", 6),
    ("type_logement", "maison endommagee", 4), ("type_logement", "maison correcte", 0),
    ("articles_essentiels", "0 ou 1 sur 5", 8), ("articles_essentiels", "2 ou 3 sur 5", 4), ("articles_essentiels", "4 ou 5 sur 5", 0),
    ("repas_par_jour", "1 ou moins", 12), ("repas_par_jour", "2", 5), ("repas_par_jour", "3", 0),
    ("jours_sans_nourriture_7j", "3 jours ou plus", 8), ("jours_sans_nourriture_7j", "1 ou 2 jours", 4),
    ("source_revenu", "aucun revenu / mendicite", 12), ("source_revenu", "travail journalier", 6),
    ("source_revenu", "agriculture ou petit commerce", 3), ("source_revenu", "salaire regulier", 0),
    ("possede_betail", "oui", -5),
]
INCLUSIONS = ["chef de ménage mineur", "enfant non accompagné hébergé ET chef de ménage femme ou âgé"]
EXCLUSIONS = ["salaire régulier", "vivres ou cash reçus dans les 3 derniers mois (anti-doublon)"]


def saute(garder, valeurs) -> pd.Series:
    """Question non posée (logique de saut Kobo) : la cellule exportée est vide."""
    return pd.Series(valeurs, dtype=object).where(np.asarray(garder))


def tirage(rng, modalites, *probas) -> np.ndarray:
    n = max(np.size(p) for p in probas)
    p = np.column_stack([np.broadcast_to(np.clip(x, 0, None), (n,)) for x in probas])
    p = p / p.sum(1, keepdims=True)
    return np.asarray(modalites)[(rng.random(n)[:, None] > p.cumsum(1)).sum(1).clip(0, len(modalites) - 1)]


def generer(n: int = N_MENAGES, graine: int = GRAINE) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    d = pd.DataFrame({"id_menage": [f"MEN_{i:06d}" for i in range(1, n + 1)]})
    d["date_enquete"] = (pd.Timestamp("2026-01-05") + pd.to_timedelta(rng.integers(0, 240, n), unit="D")).strftime("%Y-%m-%d")
    d["enqueteur"] = rng.choice([f"ENQ_{i:02d}" for i in range(1, 21)], n)
    d["localite"] = rng.choice(list(LOCALITES), n)
    risque = d["localite"].map(LOCALITES).to_numpy()

    d["statut_menage"] = tirage(rng, ["resident", "famille_accueil", "deplace", "retourne"],
                                0.60 - 0.35 * risque, 0.12, 0.10 + 0.40 * risque, 0.10)
    deplace = (d["statut_menage"] == "deplace").to_numpy()
    d["duree_deplacement_mois"] = saute(deplace, rng.integers(0, 37, n)).astype(float)
    # vulnérabilité latente : relie les réponses entre elles (non exportée)
    v = np.clip(0.10 + 0.30 * risque + 0.25 * deplace + 0.10 * (d["statut_menage"] == "retourne") + rng.normal(0, 0.15, n), 0, 1)

    d["sexe_chef_menage"] = np.where(rng.random(n) < 0.28 + 0.12 * deplace, "femme", "homme")
    age = np.clip(rng.normal(41, 14, n), 18, 90).astype(int)
    d["age_chef_menage"] = np.where(rng.random(n) < 0.015 + 0.02 * deplace, rng.integers(14, 18, n), age)
    taille = np.clip(1 + rng.poisson(5.0, n), 1, 18)
    d["taille_menage"] = taille
    reste = taille - 1
    groupes = np.array([rng.multinomial(k, [0.22, 0.34, 0.37, 0.07]) for k in reste])  # 0-4, 5-17, 18-59, 60+
    chef = np.select([d["age_chef_menage"] < 18, d["age_chef_menage"] >= 60], [1, 3], 2)
    groupes[np.arange(n), chef] += 1
    d["enfants_moins_5"], d["enfants_5_17"], d["adultes_18_59"], d["personnes_60_plus"] = groupes.T
    d["niveau_instruction_chef"] = rng.choice(["aucun", "primaire", "secondaire", "superieur"], n, p=[0.27, 0.43, 0.25, 0.05])
    d["membre_handicap"] = np.where(rng.random(n) < 0.08 + 0.06 * v, "oui", "non")
    d["membre_maladie_chronique"] = np.where(rng.random(n) < 0.10 + 0.06 * v, "oui", "non")
    d["femme_enceinte_allaitante"] = np.where(rng.random(n) < 0.20, "oui", "non")
    d["enfant_non_accompagne_heberge"] = np.where(rng.random(n) < 0.01 + 0.06 * v * (1 + deplace), "oui", "non")

    d["type_logement"] = tirage(rng, ["maison_correcte", "maison_endommagee", "heberge_famille", "abri_fortune", "sans_abri"],
                                0.55 - 0.45 * v, 0.20, 0.05 + 0.15 * deplace, 0.03 + 0.25 * v * deplace + 0.05 * v, 0.005 + 0.03 * v)
    d["nombre_pieces"] = np.clip(rng.poisson(2.6 - v, n) + 1, 1, 8)
    d["articles_essentiels"] = rng.binomial(5, np.clip(0.85 - 0.6 * v, 0.05, 0.95))
    d["repas_par_jour"] = tirage(rng, [3, 2, 1], 0.55 - 0.45 * v, 0.35, 0.03 + 0.30 * v).astype(int)
    d["jours_sans_nourriture_7j"] = np.clip(rng.binomial(7, np.clip(0.01 + 0.25 * v ** 2, 0, 1)), 0, 7)
    d["source_revenu"] = tirage(rng, ["salaire_regulier", "petit_commerce", "agriculture", "travail_journalier", "aucun_revenu"],
                                0.10 - 0.07 * v, 0.20, 0.35, 0.20 + 0.10 * v, 0.03 + 0.15 * v)
    d["possede_betail"] = np.where(rng.random(n) < 0.35 - 0.25 * v, "oui", "non")
    d["possede_telephone"] = np.where(rng.random(n) < 0.70 - 0.30 * v, "oui", "non")
    d["distance_marche_min"] = np.clip(rng.gamma(2, 20 + 25 * v), 5, 300).astype(int)

    recu = rng.random(n) < 0.12 + 0.20 * v
    d["assistance_3_derniers_mois"] = np.where(recu, "oui", "non")
    d["type_assistance_recue"] = saute(recu, rng.choice(["vivres", "cash", "articles_menagers", "abri", "autre"], n,
                                                        p=[0.35, 0.25, 0.20, 0.10, 0.10]))

    d["score_excel"], d["motif_decision"], d["eligible"] = feuille_excel(d)
    return d


def points(d: pd.DataFrame) -> pd.DataFrame:
    """Points de chaque critère, comme les colonnes intermédiaires de la feuille Excel."""
    duree = d["duree_deplacement_mois"].fillna(99)
    p = pd.DataFrame(index=d.index)
    p["statut_menage"] = np.select(
        [(d["statut_menage"] == "deplace") & (duree < 3), d["statut_menage"] == "deplace",
         d["statut_menage"] == "retourne", d["statut_menage"] == "famille_accueil"], [15, 10, 8, 5], 0)
    p["chef_menage"] = (6 * (d["sexe_chef_menage"] == "femme") + 12 * (d["age_chef_menage"] < 18)
                        + 8 * (d["age_chef_menage"] >= 60))
    p["taille_menage"] = np.select([d["taille_menage"] >= 10, d["taille_menage"] >= 7, d["taille_menage"] >= 4], [10, 7, 4], 0)
    p["enfants_moins_5"] = np.minimum(2 * d["enfants_moins_5"], 6)
    p["membre_handicap"] = 8 * (d["membre_handicap"] == "oui")
    p["membre_maladie_chronique"] = 5 * (d["membre_maladie_chronique"] == "oui")
    p["femme_enceinte_allaitante"] = 5 * (d["femme_enceinte_allaitante"] == "oui")
    p["enfant_non_accompagne_heberge"] = 8 * (d["enfant_non_accompagne_heberge"] == "oui")
    p["type_logement"] = d["type_logement"].map({"sans_abri": 15, "abri_fortune": 10, "heberge_famille": 6,
                                                 "maison_endommagee": 4, "maison_correcte": 0})
    p["articles_essentiels"] = np.select([d["articles_essentiels"] <= 1, d["articles_essentiels"] <= 3], [8, 4], 0)
    p["repas_par_jour"] = d["repas_par_jour"].map({1: 12, 2: 5, 3: 0})
    p["jours_sans_nourriture_7j"] = np.select([d["jours_sans_nourriture_7j"] >= 3, d["jours_sans_nourriture_7j"] >= 1], [8, 4], 0)
    p["source_revenu"] = d["source_revenu"].map({"aucun_revenu": 12, "travail_journalier": 6, "agriculture": 3,
                                                 "petit_commerce": 3, "salaire_regulier": 0})
    p["possede_betail"] = -5 * (d["possede_betail"] == "oui")
    return p.astype(int)


def feuille_excel(d: pd.DataFrame):
    score = points(d).sum(axis=1)
    inclusion = (d["age_chef_menage"] < 18) | ((d["enfant_non_accompagne_heberge"] == "oui")
                                                & ((d["sexe_chef_menage"] == "femme") | (d["age_chef_menage"] >= 60)))
    excl_salaire = d["source_revenu"] == "salaire_regulier"
    excl_doublon = d["type_assistance_recue"].isin(["vivres", "cash"])
    motif = np.select(
        [excl_salaire, excl_doublon, inclusion, score >= SEUIL_ELIGIBILITE],
        ["exclusion : salaire régulier", "exclusion : assistance récente (vivres/cash)",
         "inclusion automatique", f"score >= {SEUIL_ELIGIBILITE}"], f"score < {SEUIL_ELIGIBILITE}")
    eligible = (~excl_salaire & ~excl_doublon & (inclusion | (score >= SEUIL_ELIGIBILITE))).astype(int)
    return score, motif, eligible


def classement_excel(d: pd.DataFrame) -> pd.Series:
    """Rang de priorité de l'agent parmi les éligibles : score décroissant, puis ménage le plus grand, puis date."""
    elig = d[d["eligible"] == 1].sort_values(["score_excel", "taille_menage", "date_enquete"], ascending=[False, False, True])
    return pd.Series(np.arange(1, len(elig) + 1), index=elig.index, name="rang_excel")


def construire_base(d: pd.DataFrame, chemin_db: Path, chemin_csv: Path) -> None:
    chemin_db.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(chemin_csv, index=False, encoding="utf-8")
    if chemin_db.exists():
        chemin_db.unlink()
    dico = pd.DataFrame({"variable": d.columns, "type": [str(t) for t in d.dtypes],
                         "cellules_vides_pct": (d.isna().mean() * 100).round(1).to_numpy(),
                         "exemple": [str(d[c].dropna().iloc[0]) for c in d.columns]})
    regles = pd.DataFrame({"type": ["inclusion automatique"] * len(INCLUSIONS) + ["exclusion"] * len(EXCLUSIONS)
                           + ["seuil"], "regle": INCLUSIONS + EXCLUSIONS + [f"éligible si score >= {SEUIL_ELIGIBILITE}"]})
    with sqlite3.connect(chemin_db) as con:
        d.to_sql("menages", con, index=False)
        pd.DataFrame(GRILLE, columns=["critere", "reponse", "points"]).to_sql("grille", con, index=False)
        regles.to_sql("regles", con, index=False)
        dico.to_sql("dictionnaire", con, index=False)
        con.execute("CREATE UNIQUE INDEX ix_menages_id ON menages(id_menage)")
