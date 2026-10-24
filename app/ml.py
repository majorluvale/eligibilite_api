"""Chargement du modèle, prédiction et explicabilité.

Un seul modèle XGBoost, entraîné sur les décisions de la vérification approfondie à partir
des réponses de l'enquête ménage (voir notebooks/ciblage_avec_ml_v6.ipynb). La grille Excel
de l'ONG, son score et son seuil ne sont pas donnés au modèle.

Préparation identique au notebook : questions conditionnelles non posées -> -1 (nombres) ou
"non_applicable" (choix), puis codage des choix comme le LabelEncoder (ordre alphabétique).

Explication : contributions TreeSHAP de XGBoost, en log-odds. Elles s'additionnent exactement :
log-odds = valeur de base + somme des contributions des questions, et probabilité = sigmoïde(log-odds).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .schema import CATEGORICAL, FEATURE_NAMES, NUMERIC, SHORT_LABELS, THEMES, libelle

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
NON_APPLICABLE_CHOIX = "non_applicable"
NON_APPLICABLE_NOMBRE = -1.0


def prepare(rows) -> pd.DataFrame:
    """Met les réponses au format des données d'entraînement (question non posée -> valeur « non applicable »)."""
    df = pd.DataFrame(rows).reindex(columns=FEATURE_NAMES)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float).fillna(NON_APPLICABLE_NOMBRE)
    for c in CATEGORICAL:
        df[c] = df[c].astype(object).where(df[c].notna(), NON_APPLICABLE_CHOIX)
    return df


def encoder(df: pd.DataFrame, encodeurs: dict[str, list[str]]) -> np.ndarray:
    """Matrice numérique dans l'ordre de FEATURE_NAMES (choix -> rang dans la liste triée du LabelEncoder)."""
    X = df[FEATURE_NAMES].copy()
    for c in CATEGORICAL:
        codes = {v: i for i, v in enumerate(encodeurs[c])}
        inconnus = set(X[c]) - set(codes)
        if inconnus:
            raise ValueError(f"{c}: modalité inconnue du modèle {sorted(map(str, inconnus))}")
        X[c] = X[c].map(codes)
    return X.to_numpy(dtype=float)


class EligibilityModel:
    priority_method = "log_odds"

    def __init__(self, artifact: dict):
        self.modele = artifact["modele"]
        self.encodeurs = artifact["encodeurs"]
        self.threshold = float(artifact["seuil"])
        self.metadata = artifact["metadata"]
        self.zone_grise = tuple(self.metadata.get("zone_grise", {}).get("probabilite", (0.2, 0.8)))
        self._booster = self.modele.get_booster()

    @property
    def version(self) -> str:
        return self.metadata["version"]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilité d'éligibilité."""
        return self.modele.predict_proba(encoder(df, self.encodeurs))[:, 1]

    def log_odds(self, df: pd.DataFrame) -> np.ndarray:
        return self.modele.predict(encoder(df, self.encodeurs), output_margin=True)

    def contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(contribution de chaque question [n, q], valeur de base [n]) en log-odds : base + somme = log-odds."""
        brut = self._booster.predict(xgb.DMatrix(encoder(df, self.encodeurs)), pred_contribs=True)
        return brut[:, :-1], brut[:, -1]

    def priorities(self, df: pd.DataFrame) -> np.ndarray:
        """Classement des éligibles sous contrainte budgétaire : log-odds du modèle."""
        return self.log_odds(df)

    def a_verifier(self, proba):
        """Zone grise : le modèle hésite (probabilité entre les deux bornes)."""
        bas, haut = self.zone_grise
        return (proba > bas) & (proba < haut)

    def predict(self, features: dict) -> dict:
        df = prepare([features])
        contrib, base = self.contributions(df)
        log_odds = float(base[0] + contrib[0].sum())
        proba = float(1 / (1 + np.exp(-log_odds)))
        eligible = proba >= self.threshold

        explanation = [{"feature": name, "label": SHORT_LABELS[name], "theme": THEMES[name],
                        "value": libelle(name, features.get(name)), "contribution": round(float(contrib[0, j]), 4)}
                       for j, name in enumerate(FEATURE_NAMES)]
        explanation.sort(key=lambda e: abs(e["contribution"]), reverse=True)
        total = sum(abs(e["contribution"]) for e in explanation) or 1.0
        for e in explanation:
            e["share"] = round(abs(e["contribution"]) / total, 4)

        return {
            "eligible": bool(eligible),
            "score": round(proba, 4),
            "threshold": round(self.threshold, 4),
            "log_odds": round(log_odds, 4),
            "a_verifier": bool(self.a_verifier(proba)),
            "base_value": round(float(base[0]), 4),
            "explanation": explanation,
            "themes": par_theme(explanation),
            "summary": summarize(explanation, eligible, proba, self.threshold),
            "model_version": self.version,
            "indice_priorite": round(log_odds, 4),
            "methode_priorite": self.priority_method,
        }


def par_theme(explanation: list[dict]) -> list[dict]:
    """Somme des contributions par thème du questionnaire, du plus influent au moins influent."""
    themes: dict[str, float] = {}
    for e in explanation:
        themes[e["theme"]] = themes.get(e["theme"], 0.0) + e["contribution"]
    return sorted(({"theme": t, "contribution": round(v, 4)} for t, v in themes.items()),
                  key=lambda t: abs(t["contribution"]), reverse=True)


def summarize(explanation: list[dict], eligible: bool, proba: float, seuil: float) -> str:
    fmt = lambda items: ", ".join(f"{e['label']} = {e['value']} ({e['contribution']:+.2f})" for e in items)
    pour = [e for e in explanation if e["contribution"] >= 0.1][:4]
    contre = [e for e in explanation if e["contribution"] <= -0.1][:4]
    parts = [f"Probabilité d'éligibilité : {proba:.0%} (seuil : {seuil:.0%}).",
             f"Décision du modèle : {'ÉLIGIBLE' if eligible else 'NON ÉLIGIBLE'}."]
    if pour:
        parts.append(f"Réponses qui rapprochent de l'éligibilité : {fmt(pour)}.")
    if contre:
        parts.append(f"Réponses qui en éloignent : {fmt(contre)}.")
    return " ".join(parts)


@lru_cache
def load_model(model_dir: str | None = None) -> EligibilityModel:
    path = Path(model_dir) if model_dir else MODEL_DIR
    return EligibilityModel(joblib.load(path / "model.joblib"))


def load_metadata(model_dir: str | None = None) -> dict:
    path = Path(model_dir) if model_dir else MODEL_DIR
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))
