"""Indicateurs standard calculés à partir des réponses du ménage (modèle du type d'assistance).

Ce ne sont pas des données externes : chaque indicateur est un calcul classique de
l'humanitaire fait sur les réponses de l'enquête, comme dans notebooks/ciblage_avec_ml.ipynb (partie 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INDICATEURS = {
    "fcs": "Score de consommation alimentaire (FCS)",
    "rcsi": "Indice réduit des stratégies d'adaptation (rCSI)",
    "revenu_par_personne": "Revenu par personne (USD, 30 j)",
    "part_non_scolarises": "Part des enfants de 5-17 ans non scolarisés",
    "dette_sur_revenu": "Dette rapportée au revenu",
}


def calculer(df: pd.DataFrame) -> pd.DataFrame:
    """Une colonne par indicateur (ordre de INDICATEURS)."""
    out = pd.DataFrame(index=df.index)
    # FCS : poids du Programme alimentaire mondial par groupe d'aliments
    out["fcs"] = (2 * df["jours_cereales"] + 3 * df["jours_legumineuses"] + 4 * df["jours_lait"] + 4 * df["jours_viande_poisson"]
                  + df["jours_legumes"] + df["jours_fruits"] + 0.5 * df["jours_huile"] + 0.5 * df["jours_sucre"])
    out["rcsi"] = (df["rcsi_aliments_moins_chers"] + 2 * df["rcsi_emprunt_nourriture"] + df["rcsi_reduction_portions"]
                   + 3 * df["rcsi_adultes_pour_enfants"] + df["rcsi_moins_de_repas"])
    out["revenu_par_personne"] = df["revenu_30j_usd"] / df["taille_menage"]
    out["part_non_scolarises"] = df["enfants_non_scolarises"] / np.maximum(df["enfants_5_17"], 1)
    out["dette_sur_revenu"] = df["dette_usd"] / (df["revenu_30j_usd"] + 20)
    return out.astype(float)
