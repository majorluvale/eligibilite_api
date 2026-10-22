"""Entraînement reproductible des modèles déployés (données v5 : grille de points d'une ONG).

Usage :
    python -m training.train                       # données v5 (générées si absentes)
    python -m training.train --data mes_listes.csv --seuil-grille 40

Le fichier doit contenir les questions de app/schema.py, la décision de l'agent
(`eligible`, 0/1) et le score qu'il a calculé avec sa grille (`score_excel`).
Voir notebooks/ciblage_avec_ml_v5.ipynb pour l'analyse complète.

Deux modèles :
1. modèle de score : prédit `score_excel` (régression LightGBM) ;
2. modèle de décision : prédit `eligible` à partir des réponses et du score prédit
   (stacking ; le score du train est calculé hors-pli pour éviter toute fuite).
Seuil de décision : meilleur F1 parmi les seuils qui garantissent une précision >= 98 %
sur la validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score)
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from app.ml import SCORE, EligibilityModel, prepare  # noqa: E402
from app.schema import CATEGORICAL, FEATURE_NAMES, NUMERIC  # noqa: E402

DONNEES_V5 = ROOT / "notebooks" / "data" / "ciblage_v5.csv"
SEED = 42
PRECISION_MIN = 0.98


def preprocesseur(num=NUMERIC) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", Pipeline([("vide", SimpleImputer(strategy="constant", fill_value="non_applicable")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
        ("num", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), list(num)),
    ])


def modele_score() -> Pipeline:
    return Pipeline([("pre", preprocesseur()), ("reg", LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                                                      random_state=SEED, verbose=-1))])


def modele_decision(avec_score=True) -> Pipeline:
    num = NUMERIC + [SCORE] if avec_score else NUMERIC
    return Pipeline([("pre", preprocesseur(num)), ("clf", LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=31,
                                                                         random_state=SEED, verbose=-1))])


def charger(chemin: Path) -> pd.DataFrame:
    if not chemin.exists() and chemin == DONNEES_V5:
        import generateur_ciblage_v5 as gen
        print(f"{chemin} absent : génération des données v5…")
        gen.construire_base(gen.generer(), chemin.with_suffix(".db"), chemin)
    return pd.read_csv(chemin, encoding="utf-8-sig", keep_default_na=False, na_values=[""])


def choisir_seuil(y, p) -> float:
    grille = np.round(np.arange(0.05, 0.96, 0.01), 2)
    prec = np.array([precision_score(y, p >= t, zero_division=0) for t in grille])
    f1 = np.array([f1_score(y, p >= t) for t in grille])
    ok = prec >= PRECISION_MIN
    return float(grille[ok][np.argmax(f1[ok])] if ok.any() else grille[np.argmax(f1)])


def mesures(y, p, seuil) -> dict:
    pred = p >= seuil
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"seuil": seuil, "precision": round(precision_score(y, pred), 4), "rappel": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4), "accuracy": round(accuracy_score(y, pred), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "matrice_confusion": {"vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)}}


def recouvrement_top(priorite, reference, part=0.30) -> float:
    """Part de la liste de l'agent (top `part` des éligibles par score) retrouvée par un autre classement."""
    n = max(1, int(len(reference) * part))
    return len(set(np.argsort(-priorite, kind="stable")[:n]) & set(np.argsort(-reference, kind="stable")[:n])) / n


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DONNEES_V5)
    ap.add_argument("--out", type=Path, default=ROOT / "model")
    ap.add_argument("--seuil-grille", type=float, default=40, help="seuil d'éligibilité de la grille de l'ONG (points)")
    ap.add_argument("--marge-points", type=float, default=3, help="zone grise : score à moins de N points du seuil")
    args = ap.parse_args(argv)

    df = charger(args.data)
    X, y, s = prepare(df), df["eligible"].to_numpy(), df["score_excel"].to_numpy(dtype=float)
    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    X_tr, X_va, X_te = X.iloc[i_tr], X.iloc[i_va], X.iloc[i_te]
    print(f"Données : {len(df):,} ménages, {y.mean():.1%} éligibles | train {len(i_tr):,} / val {len(i_va):,} / test {len(i_te):,}")

    # 1. modèle de score, puis score hors-pli pour entraîner la décision sans fuite
    score_pipe = modele_score().fit(X_tr, s[i_tr])
    oof = cross_val_predict(modele_score(), X_tr, s[i_tr], cv=KFold(5, shuffle=True, random_state=SEED))
    decision_pipe = modele_decision().fit(X_tr.assign(**{SCORE: oof}), y[i_tr])
    artifact = {"score_pipeline": score_pipe, "decision_pipeline": decision_pipe, "threshold": 0.5,
                "metadata": {"version": "", "seuil_grille": args.seuil_grille, "zone_grise": {"marge_points": args.marge_points}}}
    modele = EligibilityModel(artifact)

    # comparaison sur la validation (seuil 0,5)
    lr = Pipeline([("pre", preprocesseur()), ("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]).fit(X_tr, y[i_tr])
    direct = modele_decision(avec_score=False).fit(X_tr, y[i_tr])
    p_va = modele.score(X_va)
    comparaison = []
    for nom, p in [("regression_logistique", lr.predict_proba(X_va)[:, 1]), ("lightgbm", direct.predict_proba(X_va)[:, 1]),
                   ("score_predit_plus_lightgbm", p_va)]:
        comparaison.append({"modele": nom, **mesures(y[i_va], p, 0.5)})
        print(f"  {nom:28s} précision={comparaison[-1]['precision']:.4f} rappel={comparaison[-1]['rappel']:.4f}")

    seuil = choisir_seuil(y[i_va], p_va)
    artifact["threshold"] = seuil
    p_te = modele.score(X_te)
    s_te = modele.points(X_te)
    test = {**mesures(y[i_te], p_te, seuil), "erreur_score_points": round(float(mean_absolute_error(s[i_te], s_te)), 3)}
    print(f"Seuil {seuil:.2f} — test : {json.dumps(test, ensure_ascii=False)}")

    erreurs = (p_te >= seuil) != y[i_te].astype(bool)
    a_verifier = modele.a_verifier(s_te, p_te)
    zone_grise = {"marge_points": args.marge_points,
                  "part_menages": round(float(a_verifier.mean()), 4),
                  "part_erreurs": round(float((a_verifier & erreurs).sum() / max(erreurs.sum(), 1)), 4)}

    accord = (pd.DataFrame({"motif": df["motif_decision"].iloc[i_te].to_numpy(), "ok": ~erreurs})
              .groupby("motif").ok.agg(["mean", "size"]).reset_index()) if "motif_decision" in df else pd.DataFrame()

    elig = y[i_te] == 1
    log_odds = decision_pipe.steps[-1][1].predict(
        decision_pipe.named_steps["pre"].transform(modele._avec_score(X_te[elig])), raw_score=True)
    priorite = {"methode": "score_predit", "cible": "score_excel",
                "erreur_moyenne_points": test["erreur_score_points"],
                "spearman": round(float(spearmanr(s_te[elig], s[i_te][elig]).correlation), 4),
                "recouvrement_top30_score_predit": round(recouvrement_top(s_te[elig], s[i_te][elig]), 4),
                "recouvrement_top30_log_odds": round(recouvrement_top(log_odds, s[i_te][elig]), 4)}
    print(f"Priorité : {json.dumps(priorite, ensure_ascii=False)}")

    contrib, _ = modele.point_contributions(X_te.iloc[:5000])
    importance = sorted([{"feature": f, "importance": round(float(v), 3)} for f, v in zip(FEATURE_NAMES, np.abs(contrib).mean(0))],
                        key=lambda r: r["importance"], reverse=True)

    metadata = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "modele": "score prédit + LightGBM",
        "seuil": seuil, "seuil_grille": args.seuil_grille, "precision_minimale": PRECISION_MIN,
        "taux_eligibles_entrainement": round(float(y.mean()), 4), "n_entrainement": len(i_tr),
        "donnees": str(args.data.name),
        "validation_comparaison": comparaison, "test": test,
        "accord_par_motif": [{"motif": r.motif, "accord": round(float(r["mean"]), 4), "menages": int(r["size"])}
                             for _, r in accord.iterrows()],
        "zone_grise": zone_grise, "priorite": priorite, "importance_globale": importance, "variables": FEATURE_NAMES,
    }
    artifact["metadata"] = metadata
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out / "model.joblib")
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artefacts écrits dans {args.out}")
    return metadata


if __name__ == "__main__":
    main()
