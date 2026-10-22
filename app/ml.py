"""Chargement des modèles, prédiction et explicabilité.

Deux modèles, à l'image du travail de l'agent :
1. le **modèle de score** (régression) estime le score que l'agent calculerait avec la
   grille de l'ONG. Ses contributions TreeSHAP s'additionnent exactement :
   score estimé = base + somme des points apportés par chaque question ;
2. le **modèle de décision** (classification) reçoit les réponses et le score estimé,
   et décide éligible ou non. Il apprend aussi les exceptions de la grille
   (exclusions, inclusions automatiques) que le score seul ne capture pas.
Le score estimé sert aussi à classer les éligibles quand le budget est limité.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .schema import CATEGORICAL, FEATURE_NAMES, NUMERIC, SHORT_LABELS, libelle

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SCORE = "score_predit"


def prepare(rows) -> pd.DataFrame:
    """Met les réponses au format des données d'entraînement (question non posée -> NaN)."""
    df = pd.DataFrame(rows).reindex(columns=FEATURE_NAMES)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)  # même type qu'à l'entraînement (NaN possibles)
    for c in CATEGORICAL:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def column_sources(transformed_names, extra=()) -> list[str]:
    """'cat__type_logement_abri_fortune' -> 'type_logement', 'num__missingindicator_x' -> 'x'."""
    connues = sorted(list(FEATURE_NAMES) + list(extra), key=len, reverse=True)
    sources = []
    for col in transformed_names:
        nom = col.split("__", 1)[1].replace("missingindicator_", "")
        sources.append(next(q for q in connues if nom == q or nom.startswith(q + "_")))
    return sources


def _aggregation(pipeline, questions) -> np.ndarray:
    noms = pipeline.named_steps["pre"].get_feature_names_out()
    agg = np.zeros((len(noms), len(questions)))
    for i, q in enumerate(column_sources(noms, extra=[SCORE] if SCORE in questions else [])):
        agg[i, questions.index(q)] = 1.0
    return agg


class EligibilityModel:
    priority_method = "score_predit"

    def __init__(self, artifact: dict):
        self.score_pipeline = artifact["score_pipeline"]
        self.decision_pipeline = artifact["decision_pipeline"]
        self.threshold = float(artifact["threshold"])
        self.metadata = artifact["metadata"]
        self.seuil_grille = float(self.metadata["seuil_grille"])
        self.marge_points = float(self.metadata.get("zone_grise", {}).get("marge_points", 3))
        self._agg_score = _aggregation(self.score_pipeline, list(FEATURE_NAMES))
        self._questions_decision = list(FEATURE_NAMES) + [SCORE]
        self._agg_decision = _aggregation(self.decision_pipeline, self._questions_decision)

    @property
    def version(self) -> str:
        return self.metadata["version"]

    # --------------------------------------------------------------- score en points
    def points(self, df: pd.DataFrame) -> np.ndarray:
        return self.score_pipeline.predict(df[FEATURE_NAMES])

    def point_contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(points apportés par chaque question [n, q], valeur de base [n]) : base + somme = score estimé."""
        pre, reg = self.score_pipeline.named_steps["pre"], self.score_pipeline.steps[-1][1]
        brut = reg.predict(pre.transform(df[FEATURE_NAMES]), pred_contrib=True)
        return brut[:, :-1] @ self._agg_score, brut[:, -1]

    def priorities(self, df: pd.DataFrame) -> np.ndarray:
        """Classement des éligibles sous contrainte budgétaire : score estimé, en points."""
        return self.points(df)

    # --------------------------------------------------------------- décision
    def _avec_score(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[FEATURE_NAMES].assign(**{SCORE: self.points(df)})

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilité d'éligibilité donnée par le modèle de décision."""
        return self.decision_pipeline.predict_proba(self._avec_score(df))[:, 1]

    def decision_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contributions (log-odds) du modèle de décision par question, score estimé compris."""
        pre, clf = self.decision_pipeline.named_steps["pre"], self.decision_pipeline.steps[-1][1]
        brut = clf.predict_proba(pre.transform(self._avec_score(df)), pred_contrib=True)
        return pd.DataFrame(brut[:, :-1] @ self._agg_decision, columns=self._questions_decision, index=df.index)

    def a_verifier(self, points, proba):
        """Zone grise : score à quelques points du seuil de la grille, ou modèle de décision hésitant."""
        return (np.abs(points - self.seuil_grille) <= self.marge_points) | ((proba > 0.2) & (proba < 0.8))

    def predict(self, features: dict) -> dict:
        df = prepare([features])
        proba = float(self.score(df)[0])
        eligible = proba >= self.threshold
        contrib, base = self.point_contributions(df)
        points = float(base[0] + contrib[0].sum())

        explanation = [{"feature": name, "label": SHORT_LABELS[name], "value": libelle(name, features.get(name)),
                        "contribution": round(float(contrib[0, j]), 3)} for j, name in enumerate(FEATURE_NAMES)]
        explanation.sort(key=lambda e: abs(e["contribution"]), reverse=True)
        total = sum(abs(e["contribution"]) for e in explanation) or 1.0
        for e in explanation:
            e["share"] = round(abs(e["contribution"]) / total, 4)

        exception = None
        if eligible != (points >= self.seuil_grille):
            # la décision contredit le score : le modèle applique une règle d'exception de la grille
            dc = self.decision_contributions(df).iloc[0].drop(SCORE)
            q = dc.idxmax() if eligible else dc.idxmin()
            exception = {"type": "inclusion" if eligible else "exclusion",
                         "facteur": f"{SHORT_LABELS[q]} = {libelle(q, features.get(q))}"}

        return {
            "eligible": bool(eligible),
            "score": round(proba, 4),
            "threshold": round(self.threshold, 4),
            "points": round(points, 1),
            "seuil_grille": self.seuil_grille,
            "a_verifier": bool(self.a_verifier(points, proba)),
            "base_value": round(float(base[0]), 3),
            "explanation": explanation,
            "exception": exception,
            "summary": summarize(explanation, eligible, points, self.seuil_grille, exception),
            "model_version": self.version,
            "indice_priorite": round(points, 3),
            "methode_priorite": self.priority_method,
        }


def summarize(explanation: list[dict], eligible: bool, points: float, seuil: float, exception: dict | None) -> str:
    fmt = lambda items: ", ".join(f"{e['label']} = {e['value']} ({e['contribution']:+.1f})" for e in items)
    pour = [e for e in explanation if e["contribution"] >= 0.5][:3]
    contre = [e for e in explanation if e["contribution"] <= -0.5][:3]
    parts = [f"Score estimé : {points:.1f} points (seuil de la grille : {seuil:g}).",
             f"Décision du modèle : {'ÉLIGIBLE' if eligible else 'NON ÉLIGIBLE'}."]
    if exception:
        parts.append(f"Exception : le modèle applique une {exception['type']} ({exception['facteur']}) "
                     f"{'malgré un score insuffisant' if eligible else 'malgré un score suffisant'}.")
    if pour:
        parts.append(f"Critères qui apportent le plus de points : {fmt(pour)}.")
    if contre:
        parts.append(f"Critères qui en retirent : {fmt(contre)}.")
    return " ".join(parts)


@lru_cache
def load_model(model_dir: str | None = None) -> EligibilityModel:
    path = Path(model_dir) if model_dir else MODEL_DIR
    return EligibilityModel(joblib.load(path / "model.joblib"))


def load_metadata(model_dir: str | None = None) -> dict:
    path = Path(model_dir) if model_dir else MODEL_DIR
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))
