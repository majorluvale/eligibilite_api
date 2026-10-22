"""Chargement du modèle, prédiction et explicabilité.

Les explications sont des contributions additives en log-odds :
    logit(score) = valeur_de_base + somme(contributions)
- régression logistique : contribution exacte coef * (x - moyenne_entraînement)
- LightGBM : valeurs SHAP exactes (TreeSHAP natif, `pred_contrib=True`)
Les colonnes one-hot et les indicateurs dérivés sont regroupés sous la
question du formulaire d'où ils proviennent, pour que l'explication parle le
langage de l'enquêteur.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import ENGINEERED_SOURCES
from .schema import CATEGORICAL, FEATURE_NAMES, SHORT_LABELS

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def column_sources(transformed_names) -> list[str]:
    """Pour chaque colonne transformée ('cat__Location_Zone A', 'num__Chef_Mineur'...) -> question d'origine."""
    sources = []
    for col in transformed_names:
        name = col.split("__", 1)[1]
        if name in ENGINEERED_SOURCES:
            sources.append(ENGINEERED_SOURCES[name])
        elif name in FEATURE_NAMES:
            sources.append(name)
        else:
            sources.append(next(c for c in CATEGORICAL if name.startswith(c + "_")))
    return sources


class EligibilityModel:
    def __init__(self, artifact: dict):
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.background_mean = artifact.get("background_mean")
        self.metadata = artifact["metadata"]
        # modèle de priorité (score de l'agent prédit) ; absent si l'historique n'avait pas de score
        self.priority_pipeline = artifact.get("priority_pipeline")
        self.clf = self.pipeline.steps[-1][1]
        self.preprocess = self.pipeline[:-1]
        names = self.pipeline.named_steps["pre"].get_feature_names_out()
        src = column_sources(names)
        # matrice d'agrégation colonnes transformées -> questions brutes
        self.agg = np.zeros((len(names), len(FEATURE_NAMES)))
        for i, s in enumerate(src):
            self.agg[i, FEATURE_NAMES.index(s)] = 1.0

    @property
    def version(self) -> str:
        return self.metadata["version"]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[FEATURE_NAMES])[:, 1]

    def contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Retourne (contributions [n, n_questions], valeur_de_base [n]) en log-odds."""
        z = np.asarray(self.preprocess.transform(df[FEATURE_NAMES]), dtype=float)
        if hasattr(self.clf, "coef_"):
            coef = self.clf.coef_[0]
            contrib = coef * (z - self.background_mean)
            base = np.full(len(z), self.clf.intercept_[0] + coef @ self.background_mean)
        else:
            raw = self.clf.predict_proba(z, pred_contrib=True)
            contrib, base = raw[:, :-1], raw[:, -1]
        return contrib @ self.agg, base

    @property
    def priority_method(self) -> str:
        return "score_predit" if self.priority_pipeline is not None else "log_odds"

    def priorities(self, df: pd.DataFrame) -> np.ndarray:
        """Indice de priorité pour classer les éligibles sous contrainte budgétaire.

        - score_predit : score de l'agent (points de sa grille) prédit par le modèle de priorité ;
        - log_odds (repli) : base + somme des contributions du modèle de décision.
        """
        if self.priority_pipeline is not None:
            return self.priority_pipeline.predict(df[FEATURE_NAMES])
        contrib, base = self.contributions(df)
        return base + contrib.sum(axis=1)

    def predict(self, features: dict, review_band: float | None = None) -> dict:
        if review_band is None:
            review_band = self.metadata.get("zone_grise", {}).get("marge", 0.3)
        df = pd.DataFrame([features])
        score = float(self.score(df)[0])
        contrib, base = self.contributions(df)
        eligible = score >= self.threshold
        explanation = []
        for j, name in enumerate(FEATURE_NAMES):
            explanation.append({
                "feature": name,
                "label": SHORT_LABELS[name],
                "value": features[name],
                "contribution": round(float(contrib[0, j]), 4),
            })
        explanation.sort(key=lambda e: abs(e["contribution"]), reverse=True)
        total = sum(abs(e["contribution"]) for e in explanation) or 1.0
        for e in explanation:
            e["share"] = round(abs(e["contribution"]) / total, 4)
        return {
            "eligible": bool(eligible),
            "score": round(score, 4),
            "threshold": round(self.threshold, 4),
            "a_verifier": abs(score - self.threshold) <= review_band,
            "base_value": round(float(base[0]), 4),
            "explanation": explanation,
            "summary": summarize(explanation, eligible),
            "model_version": self.version,
            "indice_priorite": round(float(self.priorities(df)[0]), 4),
            "methode_priorite": self.priority_method,
        }


def summarize(explanation: list[dict], eligible: bool) -> str:
    pour = [e for e in explanation if e["contribution"] > 0.05][:3]
    contre = [e for e in explanation if e["contribution"] < -0.05][:3]
    fmt = lambda items: ", ".join(f"{e['label']} = {e['value']}" for e in items)
    parts = [f"Décision du modèle : {'ÉLIGIBLE' if eligible else 'NON ÉLIGIBLE'}."]
    if pour:
        parts.append(f"Facteurs qui rapprochent de l'éligibilité : {fmt(pour)}.")
    if contre:
        parts.append(f"Facteurs qui éloignent de l'éligibilité : {fmt(contre)}.")
    return " ".join(parts)


@lru_cache
def load_model(model_dir: str | None = None) -> EligibilityModel:
    path = Path(model_dir) if model_dir else MODEL_DIR
    return EligibilityModel(joblib.load(path / "model.joblib"))


def load_metadata(model_dir: str | None = None) -> dict:
    path = Path(model_dir) if model_dir else MODEL_DIR
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))
