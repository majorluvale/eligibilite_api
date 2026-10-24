"""Générateur de données v6 — enquête ménage complexe (72 variables) et vérification approfondie.

Processus simulé :
1. Les enquêteurs collectent l'enquête ménage avec Kobo (questions de app/schema.py) :
   profil, composition, logement, eau et hygiène, santé, consommation alimentaire,
   stratégies d'adaptation, moyens d'existence, protection, assistance reçue.
2. L'agent de ciblage calcule son score avec la grille Excel de l'ONG (colonne
   `score_grille_excel`, seuil de 40 points). La grille ne regarde qu'une quinzaine de
   questions, surtout des oui/non.
3. Une **vérification approfondie** (visite à domicile et comité de ciblage) décide si le
   ménage est réellement dans le besoin : c'est l'étiquette `eligible`. Le comité regarde
   l'ensemble du ménage : consommation alimentaire, stratégies d'adaptation, revenu par
   personne, charge de dépendance, logement, santé, protection. Ces éléments se combinent
   (un revenu faible pèse plus lourd dans un grand ménage sans réserves), ce qu'une grille
   de points additionnés ne peut pas représenter.
4. Le comité relève aussi le **besoin principal** du ménage, donc le type d'assistance idéal
   (`assistance_ideale`) : vivres, moyens d'existence, abri et articles ménagers, éducation,
   santé et nutrition, eau et hygiène, ou protection.

Le modèle apprend la décision de la vérification à partir des réponses seules : ni la grille,
ni son score, ni son seuil ne lui sont donnés. Les données sont simulées : pas de GPS ni de
données de contexte externes.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.schema import FEATURES  # noqa: E402

import generateur_ciblage_v5 as grille_v5  # même grille Excel que la v5, utilisée ici comme point de comparaison

N_MENAGES = 100_000
GRAINE = 2026
PART_ELIGIBLES = 0.30  # capacité de la vérification : environ 30 % des ménages retenus
BRUIT_COMITE = 0.04    # part de jugement du comité non expliquée par le questionnaire

LOCALITES = {"Rutshuru": 0.35, "Rwanguba": 0.50, "Binza": 0.65, "Bambo": 0.75, "Kibirizi": 0.85, "Birambizo": 0.95}


def saute(garder, valeurs) -> pd.Series:
    """Question non posée (logique de saut Kobo) : la cellule exportée est vide."""
    return pd.Series(valeurs, dtype=object).where(np.asarray(garder))


def tirage(rng, modalites, *probas) -> np.ndarray:
    n = max(np.size(p) for p in probas)
    p = np.column_stack([np.broadcast_to(np.clip(x, 0.001, None), (n,)) for x in probas])
    p = p / p.sum(1, keepdims=True)
    return np.asarray(modalites)[(rng.random(n)[:, None] > p.cumsum(1)).sum(1).clip(0, len(modalites) - 1)]


def oui(rng, p) -> np.ndarray:
    return np.where(rng.random(np.size(p)) < np.clip(p, 0, 1), "oui", "non")


def generer(n: int = N_MENAGES, graine: int = GRAINE) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    d = pd.DataFrame({"id_menage": [f"MEN_{i:06d}" for i in range(1, n + 1)]})
    d["date_enquete"] = (pd.Timestamp("2026-03-02") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")).strftime("%Y-%m-%d")
    d["enqueteur"] = rng.choice([f"ENQ_{i:02d}" for i in range(1, 21)], n)
    d["localite"] = rng.choice(list(LOCALITES), n)
    risque = d["localite"].map(LOCALITES).to_numpy()

    # ---------------------------------------------------------------- profil
    d["statut_menage"] = tirage(rng, ["resident", "famille_accueil", "deplace", "retourne"],
                                0.62 - 0.35 * risque, 0.12, 0.08 + 0.40 * risque, 0.10)
    deplace = (d["statut_menage"] == "deplace").to_numpy()
    retourne = (d["statut_menage"] == "retourne").to_numpy()
    d["duree_deplacement_mois"] = saute(deplace, np.minimum(rng.geometric(0.12, n) - 1, 120)).astype(float)
    d["nombre_deplacements_12m"] = saute(deplace | retourne, 1 + rng.poisson(0.3 + 1.2 * risque, n)).astype(float)
    # exposition à la crise et vulnérabilité latentes : relient les réponses entre elles (non exportées)
    expo = np.clip(0.15 + 0.45 * risque + 0.25 * deplace + rng.normal(0, 0.18, n), 0, 1)
    v = np.clip(0.05 + 0.25 * risque + 0.20 * deplace + 0.08 * retourne + 0.15 * expo + rng.normal(0, 0.17, n), 0, 1)

    femme = rng.random(n) < 0.25 + 0.12 * deplace
    d["sexe_chef_menage"] = np.where(femme, "femme", "homme")
    age = np.clip(rng.normal(41, 14, n), 18, 90).astype(int)
    d["age_chef_menage"] = np.where(rng.random(n) < 0.015 + 0.02 * deplace, rng.integers(14, 18, n), age)
    d["etat_civil_chef"] = np.where(femme, tirage(rng, ["marie", "celibataire", "veuf", "divorce_separe"], 0.35, 0.10, 0.35 + 0.1 * v, 0.20),
                                    tirage(rng, ["marie", "celibataire", "veuf", "divorce_separe"], 0.82, 0.10, 0.04, 0.04))
    d["instruction_chef"] = tirage(rng, ["aucun", "primaire", "secondaire", "superieur"], 0.15 + 0.3 * v, 0.43, 0.30 - 0.15 * v, 0.07 - 0.06 * v)

    # ---------------------------------------------------------------- composition
    taille = np.clip(1 + rng.poisson(5.0, n), 1, 18)
    groupes = np.array([rng.multinomial(k, [0.22, 0.34, 0.37, 0.07]) for k in taille - 1])  # 0-4, 5-17, 18-59, 60+
    chef = np.select([d["age_chef_menage"] < 18, d["age_chef_menage"] >= 60], [1, 3], 2)
    groupes[np.arange(n), chef] += 1
    d["taille_menage"] = taille
    d["enfants_moins_5"], d["enfants_5_17"], _, d["personnes_60_plus"] = groupes.T
    d["enfants_non_scolarises"] = rng.binomial(d["enfants_5_17"], np.clip(0.10 + 0.45 * v, 0, 1))
    d["femme_enceinte_allaitante"] = oui(rng, 0.20 + 0.03 * (d["enfants_moins_5"] > 0))
    d["membre_handicap"] = oui(rng, 0.07 + 0.06 * v)
    d["membre_maladie_chronique"] = oui(rng, 0.10 + 0.07 * v)
    d["enfant_non_accompagne_heberge"] = oui(rng, 0.01 + 0.05 * v * (1 + deplace))

    # ---------------------------------------------------------------- logement et biens
    d["type_logement"] = tirage(rng, ["maison_correcte", "maison_endommagee", "heberge_famille", "abri_fortune", "sans_abri"],
                                0.55 - 0.45 * v, 0.15 + 0.15 * expo, 0.05 + 0.15 * deplace, 0.03 + 0.25 * v * deplace + 0.05 * v,
                                0.005 + 0.03 * v)
    d["statut_occupation"] = np.where(deplace, tirage(rng, ["proprietaire", "locataire", "heberge_gratuit", "site_deplaces"], 0.04, 0.15, 0.36, 0.45),
                                      tirage(rng, ["proprietaire", "locataire", "heberge_gratuit", "site_deplaces"],
                                             0.62 - 0.3 * v, 0.25, 0.10 + 0.2 * v, 0.02))
    d["nombre_pieces"] = np.where(d["type_logement"] == "sans_abri", 0, np.clip(rng.poisson(np.clip(2.6 - 1.5 * v, 0.3, None)) + 1, 1, 8))
    d["articles_essentiels"] = rng.binomial(5, np.clip(0.85 - 0.6 * v, 0.05, 0.95))
    d["possede_betail"] = oui(rng, 0.35 - 0.25 * v)
    terre = rng.random(n) < 0.70 - 0.30 * v - 0.30 * deplace
    d["acces_terre_cultivable"] = np.where(terre, "oui", "non")
    d["toit_etanche"] = oui(rng, 0.85 - 0.5 * v)
    d["source_eclairage"] = tirage(rng, ["electricite", "solaire", "lampe_torche", "aucune"], 0.12 - 0.1 * v, 0.30, 0.45, 0.05 + 0.3 * v)
    d["possede_telephone"] = oui(rng, 0.70 - 0.30 * v)

    # ---------------------------------------------------------------- eau et hygiène
    d["source_eau"] = tirage(rng, ["robinet", "puits_protege", "source_non_protegee", "eau_surface"],
                             0.25 - 0.2 * v, 0.35, 0.20 + 0.2 * v, 0.05 + 0.25 * v)
    d["temps_collecte_eau_min"] = np.clip(rng.gamma(2, 8 + 25 * v), 0, 600).astype(int)
    d["type_latrine"] = tirage(rng, ["privee", "partagee", "aucune"], 0.55 - 0.4 * v, 0.30, 0.05 + 0.3 * v)
    d["savon_disponible"] = oui(rng, 0.75 - 0.45 * v)

    # ---------------------------------------------------------------- protection (déclarée par le ménage)
    d["incident_securite_3_mois"] = oui(rng, 0.03 + 0.35 * expo ** 2)
    d["biens_perdus_crise"] = oui(rng, 0.05 + 0.45 * expo * (deplace | retourne) + 0.10 * expo)
    d["membre_disparu_detenu"] = oui(rng, 0.01 + 0.08 * expo ** 2)
    d["documents_identite"] = oui(rng, 0.88 - 0.30 * deplace - 0.1 * v)
    d["restriction_mouvements"] = oui(rng, 0.05 + 0.50 * expo ** 1.5)

    # ---------------------------------------------------------------- assistance reçue
    recu = rng.random(n) < 0.12 + 0.20 * v + 0.08 * deplace
    d["assistance_3_derniers_mois"] = np.where(recu, "oui", "non")
    type_recu = rng.choice(["vivres", "cash", "articles_menagers", "abri", "autre"], n, p=[0.35, 0.25, 0.20, 0.10, 0.10])
    d["type_assistance_recue"] = saute(recu, type_recu)
    aide_alimentaire = recu & np.isin(type_recu, ["vivres", "cash"])

    # ---------------------------------------------------------------- moyens d'existence
    d["source_revenu"] = tirage(rng, ["salaire_regulier", "petit_commerce", "agriculture", "travail_journalier", "aucun_revenu"],
                                0.10 - 0.07 * v, 0.20, 0.35 * terre + 0.05, 0.20 + 0.15 * v, 0.03 + 0.18 * v)
    moyen = pd.Series(d["source_revenu"]).map({"salaire_regulier": 160, "petit_commerce": 85, "agriculture": 55,
                                               "travail_journalier": 45, "aucun_revenu": 6}).to_numpy()
    adultes = np.maximum(taille - d["enfants_moins_5"] - d["enfants_5_17"], 1)
    d["membres_avec_revenu"] = np.where(d["source_revenu"] == "aucun_revenu", 0,
                                        np.minimum(1 + rng.binomial(adultes - 1, np.clip(0.35 - 0.25 * v, 0.02, 1)), adultes))
    d["revenu_30j_usd"] = np.clip(moyen * (0.6 + 0.4 * d["membres_avec_revenu"]) * (1.1 - 0.6 * v)
                                  * np.exp(rng.normal(0, 0.35, n)), 0, 5000).astype(int)
    d["perte_revenu_6_mois"] = oui(rng, 0.08 + 0.35 * expo * v + 0.15 * deplace)
    d["epargne_disponible"] = oui(rng, 0.40 - 0.35 * v)

    # ---------------------------------------------------------------- consommation alimentaire
    # insécurité alimentaire latente : vulnérabilité, revenu par personne, réserves, aide reçue
    rpp = d["revenu_30j_usd"] / taille
    f = np.clip(0.75 * v + 0.30 * np.exp(-rpp / 8) - 0.12 * aide_alimentaire - 0.08 * terre + rng.normal(0, 0.12, n), 0, 1)
    d["stock_nourriture_jours"] = np.where(rng.random(n) < 0.25 + 0.6 * f, 0,
                                           np.clip(rng.gamma(1.5, 12 * (1 - f) + 2 + 20 * terre), 1, 365)).astype(int)
    for col, base, pente in [("jours_cereales", 0.97, 0.30), ("jours_legumineuses", 0.62, 0.45), ("jours_lait", 0.25, 0.22),
                             ("jours_viande_poisson", 0.38, 0.33), ("jours_legumes", 0.78, 0.40), ("jours_fruits", 0.30, 0.25),
                             ("jours_huile", 0.82, 0.50), ("jours_sucre", 0.50, 0.38)]:
        d[col] = rng.binomial(7, np.clip(base - pente * f + rng.normal(0, 0.05, n), 0.01, 0.99))
    d["repas_par_jour"] = tirage(rng, [3, 2, 1], 0.55 - 0.5 * f, 0.35, 0.03 + 0.35 * f).astype(int)
    d["jours_sans_nourriture_7j"] = rng.binomial(7, np.clip(0.01 + 0.30 * f ** 2, 0, 1))

    # ---------------------------------------------------------------- stratégies d'adaptation
    for col, base, pente in [("rcsi_aliments_moins_chers", 0.15, 0.70), ("rcsi_emprunt_nourriture", 0.05, 0.50),
                             ("rcsi_reduction_portions", 0.08, 0.65), ("rcsi_adultes_pour_enfants", 0.02, 0.45),
                             ("rcsi_moins_de_repas", 0.06, 0.60)]:
        d[col] = rng.binomial(7, np.clip(base + pente * f ** 1.3 + rng.normal(0, 0.05, n), 0, 0.99))
    a_enfants = d["enfants_5_17"].to_numpy() > 0
    d["lcs_emprunt_argent"] = oui(rng, 0.15 + 0.45 * f)
    d["lcs_vente_biens_menagers"] = oui(rng, 0.08 + 0.40 * f)
    d["lcs_reduction_depenses_sante"] = oui(rng, 0.10 + 0.40 * f)
    d["lcs_consommation_semences"] = oui(rng, (0.04 + 0.40 * f) * terre)
    d["lcs_retrait_enfants_ecole"] = oui(rng, (0.03 + 0.35 * f) * a_enfants)
    d["lcs_vente_derniere_femelle"] = oui(rng, 0.01 + 0.18 * f ** 2)
    d["lcs_vente_maison_terre"] = oui(rng, 0.01 + 0.10 * f ** 2)
    d["lcs_mendicite"] = oui(rng, 0.01 + 0.22 * f ** 2.5)
    d["lcs_travail_enfants"] = oui(rng, (0.02 + 0.25 * f ** 2) * a_enfants)
    d["part_depenses_alimentation_pct"] = np.clip(rng.normal(45 + 40 * f, 9), 5, 100).astype(int)
    d["dette_usd"] = np.where(rng.random(n) < 0.55 - 0.3 * f, 0, np.clip(rng.lognormal(np.log(30 + 150 * f), 0.8), 1, 20000)).astype(int)

    # ---------------------------------------------------------------- santé et nutrition
    u5 = d["enfants_moins_5"].to_numpy() > 0
    eau_sale = d["source_eau"].isin(["source_non_protegee", "eau_surface"]).to_numpy()
    d["enfant_malade_2_semaines"] = oui(rng, (0.12 + 0.25 * v + 0.12 * eau_sale) * u5)
    d["acces_soins"] = tirage(rng, ["oui", "non_cout", "non_distance", "non_insecurite", "pas_besoin"],
                              0.55 - 0.35 * v, 0.06 + 0.30 * f, 0.04 + 0.08 * v, 0.02 + 0.20 * expo ** 2, 0.25)
    d["vaccination_enfants_a_jour"] = np.where(u5, oui(rng, 0.85 - 0.40 * v - 0.15 * deplace), "pas_enfant")
    d["enfant_malnutrition_depiste"] = oui(rng, (0.02 + 0.25 * f ** 2 + 0.05 * (d["enfant_malade_2_semaines"] == "oui")) * u5)

    d["score_grille_excel"], _, d["decision_grille_excel"] = grille_v5.feuille_excel(d)
    d["besoin"] = besoin_verifie(d) + rng.normal(0, BRUIT_COMITE, n)
    d["eligible"] = (d["besoin"] >= d["besoin"].quantile(1 - PART_ELIGIBLES)).astype(int)
    d["assistance_ideale"] = assistance_ideale(d, rng)
    colonnes = ["id_menage", "date_enquete", "enqueteur"] + [f.name for f in FEATURES]
    return d[colonnes + ["score_grille_excel", "decision_grille_excel", "eligible", "assistance_ideale"]]


def indicateurs(d: pd.DataFrame) -> pd.DataFrame:
    """Indicateurs standard calculés par le comité (FCS, rCSI, niveau LCS, revenu par personne...)."""
    oui_ = lambda c: (d[c] == "oui").astype(float)
    i = pd.DataFrame(index=d.index)
    i["fcs"] = (2 * d["jours_cereales"] + 3 * d["jours_legumineuses"] + 4 * d["jours_lait"] + 4 * d["jours_viande_poisson"]
                + d["jours_legumes"] + d["jours_fruits"] + 0.5 * d["jours_huile"] + 0.5 * d["jours_sucre"])
    i["rcsi"] = (d["rcsi_aliments_moins_chers"] + 2 * d["rcsi_emprunt_nourriture"] + d["rcsi_reduction_portions"]
                 + 3 * d["rcsi_adultes_pour_enfants"] + d["rcsi_moins_de_repas"])
    stress = oui_("lcs_emprunt_argent") + oui_("lcs_vente_biens_menagers") + oui_("lcs_reduction_depenses_sante")
    crise = oui_("lcs_consommation_semences") + oui_("lcs_retrait_enfants_ecole")
    urgence = oui_("lcs_vente_derniere_femelle") + oui_("lcs_vente_maison_terre") + oui_("lcs_mendicite") + oui_("lcs_travail_enfants")
    i["lcs"] = np.select([urgence > 0, crise > 0, stress > 0], [3, 2, 1], 0)
    i["revenu_par_personne"] = d["revenu_30j_usd"] / d["taille_menage"]
    dependants = d["enfants_moins_5"] + d["enfants_5_17"] + d["personnes_60_plus"]
    i["dependance"] = dependants / np.maximum(d["taille_menage"] - dependants, 1)
    return i


def besoin_verifie(d: pd.DataFrame) -> pd.Series:
    """Appréciation du comité de vérification : combinaison non linéaire de tout le questionnaire."""
    i = indicateurs(d)
    oui_ = lambda c: (d[c] == "oui").astype(float)
    sig = lambda x: 1 / (1 + np.exp(-x))

    alimentation = sig((38 - i["fcs"]) / 6) + 0.6 * np.clip(i["rcsi"] / 25, 0, 1.4) + 0.25 * d["jours_sans_nourriture_7j"] / 7
    reserves = np.exp(-d["stock_nourriture_jours"] / 10)
    economie = sig((9 - i["revenu_par_personne"]) / 3) * (1 + 0.6 * (d["epargne_disponible"] == "non")) \
        + 0.3 * oui_("perte_revenu_6_mois") + 0.2 * np.clip(d["dette_usd"] / (d["revenu_30j_usd"] + 20), 0, 3) / 3
    adaptation = i["lcs"] / 3
    logement = d["type_logement"].map({"sans_abri": 1.0, "abri_fortune": 0.8, "heberge_famille": 0.45,
                                       "maison_endommagee": 0.35, "maison_correcte": 0.0}) \
        * (1 + 0.3 * (d["statut_occupation"] == "site_deplaces")) + 0.2 * (d["toit_etanche"] == "non") \
        + 0.15 * (5 - d["articles_essentiels"]) / 5 + 0.15 * (d["taille_menage"] / np.maximum(d["nombre_pieces"], 1) > 3)
    eau_sale = d["source_eau"].isin(["source_non_protegee", "eau_surface"])
    sante = 0.6 * oui_("enfant_malnutrition_depiste") + 0.25 * oui_("enfant_malade_2_semaines") * (1 + eau_sale) \
        + 0.25 * d["acces_soins"].isin(["non_cout", "non_distance", "non_insecurite"]) + 0.15 * (d["type_latrine"] == "aucune")
    fragilite = 0.25 * oui_("membre_handicap") + 0.15 * oui_("membre_maladie_chronique") + 0.15 * oui_("femme_enceinte_allaitante") \
        + 0.30 * oui_("enfant_non_accompagne_heberge") + 0.35 * (d["age_chef_menage"] < 18) + 0.15 * (d["age_chef_menage"] >= 60) \
        + 0.20 * ((d["sexe_chef_menage"] == "femme") & d["etat_civil_chef"].isin(["veuf", "divorce_separe", "celibataire"])) \
        + 0.15 * np.clip(i["dependance"] / 3, 0, 1)
    crise = 0.20 * oui_("incident_securite_3_mois") + 0.20 * oui_("biens_perdus_crise") + 0.30 * oui_("membre_disparu_detenu") \
        + 0.15 * oui_("restriction_mouvements") + 0.10 * (d["documents_identite"] == "non") \
        + 0.35 * (d["duree_deplacement_mois"].fillna(99) < 3) + 0.10 * np.clip(d["nombre_deplacements_12m"].fillna(0) - 1, 0, 3)
    aide = d["type_assistance_recue"].isin(["vivres", "cash"]).astype(float)

    besoin = (1.30 * alimentation * (0.6 + 0.4 * reserves) + 0.9 * economie * (1 + 0.15 * d["taille_menage"].clip(upper=12) / 6)
              + 0.55 * adaptation + 0.50 * logement + 0.45 * sante + 0.50 * fragilite + 0.55 * crise
              + 0.35 * alimentation * crise - 0.25 * aide * alimentation)
    return besoin


# Pondération des secteurs : le comité recommande l'assistance du secteur où le besoin est le plus fort
POIDS_SECTEURS = {"vivres": 0.95, "moyens_existence": 0.95, "abri_ame": 1.00, "education": 0.95,
                  "sante_nutrition": 1.08, "eau_hygiene": 1.00, "protection": 1.05}
BRUIT_SECTEURS = 0.0  # le jugement du comité est déjà dans la décision d'éligibilité


def besoins_par_secteur(d: pd.DataFrame) -> pd.DataFrame:
    """Intensité du besoin dans chaque secteur, telle que le comité l'apprécie (une colonne par type d'assistance)."""
    i = indicateurs(d)
    oui_ = lambda c: (d[c] == "oui").astype(float)
    sig = lambda x: 1 / (1 + np.exp(-x))
    aide_alimentaire = d["type_assistance_recue"].isin(["vivres", "cash"]).astype(float)
    enfants = d["enfants_5_17"].to_numpy()
    s = pd.DataFrame(index=d.index)
    s["vivres"] = sig((35 - i["fcs"]) / 5) + 0.5 * np.clip(i["rcsi"] / 25, 0, 1.4) + 0.3 * d["jours_sans_nourriture_7j"] / 7 \
        + 0.3 * np.exp(-d["stock_nourriture_jours"] / 10) - 0.5 * aide_alimentaire
    s["moyens_existence"] = sig((9 - i["revenu_par_personne"]) / 3) * (0.4 + 0.6 * (d["epargne_disponible"] == "non")) \
        + 0.5 * oui_("perte_revenu_6_mois") + 0.3 * np.clip(d["dette_usd"] / (d["revenu_30j_usd"] + 20), 0, 3) / 3 \
        + 0.4 * (oui_("lcs_consommation_semences") + oui_("lcs_vente_derniere_femelle") + oui_("lcs_vente_maison_terre")).clip(upper=1)
    s["abri_ame"] = d["type_logement"].map({"sans_abri": 1.2, "abri_fortune": 0.9, "heberge_famille": 0.4,
                                           "maison_endommagee": 0.4, "maison_correcte": 0.0}) \
        + 0.3 * (d["toit_etanche"] == "non") + 0.5 * (5 - d["articles_essentiels"]) / 5 + 0.2 * (d["statut_occupation"] == "site_deplaces")
    s["education"] = np.where(enfants > 0, 1.4 * d["enfants_non_scolarises"] / np.maximum(enfants, 1)
                              + 0.5 * oui_("lcs_retrait_enfants_ecole") + 0.4 * oui_("lcs_travail_enfants"), 0)
    s["sante_nutrition"] = 1.0 * oui_("enfant_malnutrition_depiste") + 0.45 * oui_("enfant_malade_2_semaines") \
        + 0.35 * d["acces_soins"].isin(["non_cout", "non_distance", "non_insecurite"]) + 0.3 * (d["vaccination_enfants_a_jour"] == "non") \
        + 0.25 * (oui_("membre_handicap") + oui_("membre_maladie_chronique")).clip(upper=1)
    s["eau_hygiene"] = 0.55 * d["source_eau"].isin(["source_non_protegee", "eau_surface"]) + 0.35 * np.clip(d["temps_collecte_eau_min"] / 90, 0, 1.5) \
        + 0.45 * (d["type_latrine"] == "aucune") + 0.3 * (d["savon_disponible"] == "non")
    s["protection"] = 0.5 * oui_("incident_securite_3_mois") + 0.8 * oui_("membre_disparu_detenu") + 0.7 * oui_("enfant_non_accompagne_heberge") \
        + 0.3 * oui_("biens_perdus_crise") + 0.3 * (d["documents_identite"] == "non") + 0.3 * oui_("restriction_mouvements") \
        + 0.8 * (d["age_chef_menage"] < 18)
    return s * pd.Series(POIDS_SECTEURS)


def assistance_ideale(d: pd.DataFrame, rng) -> pd.Series:
    """Type d'assistance qui répond au besoin principal du ménage, relevé par le comité pour tous les ménages visités.

    Pour un ménage éligible, c'est l'assistance à lui donner ; pour un ménage non éligible, c'est
    son besoin principal (utile pour une orientation vers un autre partenaire).
    """
    s = besoins_par_secteur(d)
    s = s + rng.normal(0, BRUIT_SECTEURS, s.shape)
    return s.idxmax(axis=1)


def construire_base(d: pd.DataFrame, chemin_db: Path, chemin_csv: Path) -> None:
    chemin_db.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(chemin_csv, index=False, encoding="utf-8")
    if chemin_db.exists():
        chemin_db.unlink()
    with sqlite3.connect(chemin_db) as con:
        d.to_sql("menages", con, index=False)
        con.execute("CREATE UNIQUE INDEX ix_menages_id ON menages(id_menage)")
