"""Entraînement reproductible du modèle d'éligibilité (classes déséquilibrées).

Usage :
    python -m training.train --data ../dataset_humanitaire_complexe_100k.csv

Choix méthodologiques (voir le notebook ciblage_avec_ml_v3_desequilibre.ipynb) :
- Score_Vuln et Assistance_Recommandee sont retirés (fuite de la cible).
- Pas de remplacement des "outliers" par la médiane : un chef de ménage de
  85 ans ou un ménage de 25 personnes sont des cas réels et vulnérables.
- Split stratifié 70/15/15, pour garder ~13.7 % d'éligibles dans chaque partie.
- Déséquilibre : pondération des classes (class_weight="balanced") +
  seuil de décision choisi sur la validation en maximisant le F-beta (beta=2 :
  exclure un ménage vulnérable coûte plus cher qu'en inclure un de trop).
- Métriques adaptées au déséquilibre : PR-AUC, rappel, précision, F2,
  balanced accuracy (l'accuracy seule est trompeuse : 86 % en prédisant
  toujours "non éligible").
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import ENGINEERED_SOURCES, add_vulnerability_flags  # noqa: E402
from app.ml import EligibilityModel  # noqa: E402
from app.schema import CATEGORICAL, FEATURE_NAMES, NUMERIC  # noqa: E402

TARGET = "Eligible"
LEAKY = ["Score_Vuln", "Assistance_Recommandee"]
# score calculé par l'agent avec sa grille (Excel) : jamais une variable d'entrée, mais la cible du modèle de priorité
PRIORITY_TARGET = "Score_Vuln"
SEED = 42


def preprocessor(scale: bool) -> list:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC + list(ENGINEERED_SOURCES)),
    ])
    steps = [("flags", FunctionTransformer(add_vulnerability_flags)), ("pre", pre)]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def candidates() -> dict[str, Pipeline]:
    return {
        "regression_logistique_ponderee": Pipeline(preprocessor(True) + [
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0))]),
        "lightgbm_pondere": Pipeline(preprocessor(False) + [
            ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   class_weight="balanced", random_state=SEED, verbose=-1))]),
        "lightgbm_non_pondere": Pipeline(preprocessor(False) + [
            ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   random_state=SEED, verbose=-1))]),
    }


def best_threshold(y, p, beta: float) -> float:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [fbeta_score(y, (p >= t).astype(int), beta=beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y, p, thr: float, beta: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "seuil": thr,
        "pr_auc": round(average_precision_score(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "rappel": round(recall_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        f"f{beta:g}": round(fbeta_score(y, pred, beta=beta), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "matrice_confusion": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
    }


def load_data(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    y = df[TARGET].to_numpy()
    X = df.drop(columns=[TARGET] + [c for c in LEAKY if c in df])[FEATURE_NAMES]
    X.attrs["score_agent"] = df[PRIORITY_TARGET].to_numpy() if PRIORITY_TARGET in df else None
    return X, y


def priority_model(X_tr, s_tr) -> Pipeline:
    """Apprend le score de l'agent : sert à classer les éligibles quand le budget est limité."""
    return Pipeline(preprocessor(False) + [
        ("reg", LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1))
    ]).fit(X_tr, s_tr)


def overlap_top(priority, reference, share=0.30) -> float:
    """Part de la liste de référence (top `share` selon le score de l'agent) retrouvée par un autre classement."""
    n = max(1, int(len(reference) * share))
    return len(set(np.argsort(-priority, kind="stable")[:n]) & set(np.argsort(-reference, kind="stable")[:n])) / n


def split(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "dataset_humanitaire_complexe_100k.csv"))
    ap.add_argument("--out", default=str(ROOT / "model"))
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--review-band", type=float, default=0.30,
                    help="marge autour du seuil où la décision est signalée 'à vérifier'")
    args = ap.parse_args(argv)

    X, y = load_data(Path(args.data))
    score_agent = X.attrs.get("score_agent")
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y)
    print(f"Données : {len(X)} ménages, {y.mean():.1%} éligibles | train {len(X_tr)} / val {len(X_va)} / test {len(X_te)}")

    comparaison, fitted = [], {}
    for name, pipe in candidates().items():
        pipe.fit(X_tr, y_tr)
        p_va = pipe.predict_proba(X_va)[:, 1]
        thr = best_threshold(y_va, p_va, args.beta)
        m = metrics(y_va, p_va, thr, args.beta)
        comparaison.append({"modele": name, **m})
        fitted[name] = (pipe, thr)
        print(f"  {name:32s} seuil={thr:.2f} PR-AUC={m['pr_auc']:.4f} rappel={m['rappel']:.3f} "
              f"precision={m['precision']:.3f} F{args.beta:g}={m[f'f{args.beta:g}']:.4f}")

    key = f"f{args.beta:g}"
    best = max(comparaison, key=lambda r: (r[key], r["pr_auc"]))["modele"]
    pipe, thr = fitted[best]
    p_te = pipe.predict_proba(X_te)[:, 1]
    test = metrics(y_te, p_te, thr, args.beta)
    # référence sans aucun traitement du déséquilibre : modèle non pondéré, seuil 0.5
    p_ref = fitted["lightgbm_non_pondere"][0].predict_proba(X_te)[:, 1]
    test_sans_traitement = metrics(y_te, p_ref, 0.5, args.beta)
    in_band = np.abs(p_te - thr) <= args.review_band
    errors = (p_te >= thr).astype(int) != y_te
    zone_grise = {"marge": args.review_band, "part_menages": round(float(in_band.mean()), 4),
                  "part_erreurs": round(float((in_band & errors).sum() / max(errors.sum(), 1)), 4)}
    print(f"Modèle retenu : {best} (seuil {thr:.2f})")
    print(f"  Test : {json.dumps(test, ensure_ascii=False)}")

    background_mean = None
    if hasattr(pipe.steps[-1][1], "coef_"):
        background_mean = np.asarray(pipe[:-1].transform(X_tr), dtype=float).mean(axis=0)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    metadata = {
        "version": version,
        "modele": best,
        "seuil": thr,
        "beta": args.beta,
        "taux_eligibles_entrainement": round(float(y.mean()), 4),
        "n_entrainement": len(X_tr),
        "validation_comparaison": comparaison,
        "test": test,
        "test_sans_traitement": test_sans_traitement,
        "zone_grise": zone_grise,
        "variables": FEATURE_NAMES,
    }
    artifact = {"pipeline": pipe, "threshold": thr, "background_mean": background_mean, "metadata": metadata}

    # modèle de priorité : seulement si l'historique contient le score de l'agent
    if score_agent is not None:
        s = pd.Series(score_agent, index=X.index)
        reg = priority_model(X_tr, s.loc[X_tr.index])
        artifact["priority_pipeline"] = reg
        elig = X_te[y_te == 1]
        s_ref = s.loc[elig.index].to_numpy()
        s_pred = reg.predict(elig)
        contrib, base = EligibilityModel(artifact).contributions(elig)
        metadata["priorite"] = {
            "methode": "score_predit", "cible": PRIORITY_TARGET,
            "erreur_moyenne_points": round(float(np.abs(s_pred - s_ref).mean()), 3),
            "spearman": round(float(spearmanr(s_pred, s_ref).correlation), 4),
            "recouvrement_top30_score_predit": round(overlap_top(s_pred, s_ref), 4),
            "recouvrement_top30_log_odds": round(overlap_top(base + contrib.sum(axis=1), s_ref), 4),
        }
        print(f"  Priorité : {json.dumps(metadata['priorite'], ensure_ascii=False)}")
    else:
        metadata["priorite"] = {"methode": "log_odds"}

    # importance globale = moyenne des |contributions| sur le jeu de test
    contrib, _ = EligibilityModel(artifact).contributions(X_te)
    imp = np.abs(contrib).mean(axis=0)
    metadata["importance_globale"] = sorted(
        [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(FEATURE_NAMES, imp)],
        key=lambda r: r["importance"], reverse=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / "model.joblib")
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artefacts écrits dans {out}")
    return metadata


if __name__ == "__main__":
    main()
