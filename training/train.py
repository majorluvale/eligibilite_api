"""Entraînement reproductible des modèles déployés (enquête ménage, vérification approfondie).

Usage :
    python -m training.train                       # données simulées (générées si absentes)
    python -m training.train --data mes_menages.csv

Le fichier doit contenir les questions de app/schema.py et la décision de la vérification
(`eligible`, 0/1). S'il contient aussi le besoin principal relevé par le comité (`assistance_ideale`,
codes de schema.ASSISTANCES), le second modèle (type d'assistance idéale) est entraîné.
La grille Excel de l'ONG n'est utilisée par aucun des deux modèles :
si les colonnes `decision_grille_excel` / `score_grille_excel` sont présentes, elles servent
seulement de point de comparaison.

Même préparation et mêmes paramètres que notebooks/ciblage_avec_ml.ipynb (partie 1) :
- questions conditionnelles non posées -> -1 (nombres) ou "non_applicable" (choix) ;
- choix codés comme le LabelEncoder (ordre alphabétique) ;
- découpage 80 % entraînement / 10 % validation / 10 % test, stratifié, random_state=42 ;
- XGBoost avec les paramètres trouvés par GridSearchCV dans le notebook, seuil 0,5.
Modèle du type d'assistance : comme la partie 2 du notebook (réponses + indicateurs
standard de app/indicateurs.py, découpage stratifié sur le type d'assistance).
Pas de normalisation : elle ne change rien pour un modèle à base d'arbres.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from app.ml import VARIABLES_ASSISTANCE, EligibilityModel, encoder, encoder_assistance, prepare  # noqa: E402
from app.schema import ASSISTANCES, CATEGORICAL, FEATURE_NAMES, FEATURES_BY_NAME, THEMES  # noqa: E402

DONNEES = ROOT / "notebooks" / "data" / "ciblage.csv"
SEED = 42
SEUIL = 0.5
ZONE_GRISE = (0.2, 0.8)
XGB_PARAMS = dict(n_estimators=3000, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED)
XGB_ASSISTANCE = dict(n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      random_state=SEED)


def charger(chemin: Path) -> pd.DataFrame:
    if not chemin.exists() and chemin == DONNEES:
        import generateur_donnees as gen
        print(f"{chemin} absent : génération des données…")
        gen.construire_base(gen.generer(), chemin.with_suffix(".db"), chemin)
    return pd.read_csv(chemin, encoding="utf-8-sig", keep_default_na=False, na_values=[""])


def mesures(y, pred, proba=None) -> dict:
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"precision": round(precision_score(y, pred), 4), "rappel": round(recall_score(y, pred), 4),
         "f1": round(f1_score(y, pred), 4), "accuracy": round(accuracy_score(y, pred), 4)}
    if proba is not None:
        m["pr_auc"] = round(average_precision_score(y, proba), 4)
    m["matrice_confusion"] = {"vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)}
    return m


def entrainer_assistance(df, X_brut, encodeurs, eligible, cv=True):
    """Second modèle : type d'assistance idéale (7 classes), entraîné sur tous les ménages visités."""
    classes = sorted(ASSISTANCES)  # = LabelEncoder().classes_
    inconnus = set(df["assistance_ideale"].dropna()) - set(classes)
    if inconnus:
        raise ValueError(f"assistance_ideale : codes inconnus {sorted(inconnus)} (attendus : {classes})")
    garder = df["assistance_ideale"].notna().to_numpy()
    X = encoder_assistance(X_brut, encodeurs)[garder]
    y = df["assistance_ideale"][garder].map({c: i for i, c in enumerate(classes)}).to_numpy()
    elig = eligible[garder] == 1
    idx = np.arange(len(y))
    i_tr, i_vt = train_test_split(idx, train_size=0.8, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_vt, train_size=0.5, stratify=y[i_vt], random_state=SEED)

    modele = XGBClassifier(**XGB_ASSISTANCE).fit(X[i_tr], y[i_tr])
    proba = modele.predict_proba(X[i_te])
    pred = proba.argmax(1)
    yt = y[i_te]
    multi = lambda yy, pp: {"accuracy": round(accuracy_score(yy, pp), 4), "f1_macro": round(f1_score(yy, pp, average="macro"), 4),
                            "rappel_macro": round(recall_score(yy, pp, average="macro"), 4),
                            "precision_macro": round(precision_score(yy, pp, average="macro"), 4)}
    e = elig[i_te]
    resultat = {
        "classes": classes, "n_entrainement": len(i_tr), "parametres": {k: v for k, v in XGB_ASSISTANCE.items() if k != "random_state"},
        "variables": VARIABLES_ASSISTANCE,
        "repartition": {c: round(float((y == i).mean()), 4) for i, c in enumerate(classes)},
        "test": multi(yt, pred), "test_eligibles": multi(yt[e], pred[e]),
        "f1_par_classe": {c: round(float(v), 4) for c, v in zip(classes, f1_score(yt, pred, average=None))},
        "dans_les_2_propositions": round(float((np.argsort(-proba, 1)[:, :2] == yt[:, None]).any(1).mean()), 4),
        "matrice_confusion": confusion_matrix(yt, pred).tolist(),
        "validation_croisee": None,
    }
    print(f"Assistance : test {json.dumps(resultat['test'])} | éligibles {json.dumps(resultat['test_eligibles'])}")
    if cv:
        r = cross_validate(XGBClassifier(**XGB_ASSISTANCE), X[i_tr], y[i_tr], cv=5,
                           scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
        resultat["validation_croisee"] = {m: {"moyenne": round(float(r[f"test_{m}"].mean()), 4), "min": round(float(r[f"test_{m}"].min()), 4)}
                                          for m in ["accuracy", "f1_macro"]}
        print(f"Assistance, validation croisée : {json.dumps(resultat['validation_croisee'])}")
    return modele, classes, resultat


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DONNEES)
    ap.add_argument("--out", type=Path, default=ROOT / "model")
    ap.add_argument("--sans-cv", action="store_true", help="ne pas faire la validation croisée (plus rapide)")
    args = ap.parse_args(argv)

    df = charger(args.data)
    X_brut, y = prepare(df), df["eligible"].to_numpy()
    # = LabelEncoder().classes_ ; les choix du formulaire jamais observés sont ajoutés pour que l'API les accepte
    encodeurs = {c: sorted(set(X_brut[c].astype(str)) | set(FEATURES_BY_NAME[c].choices)) for c in CATEGORICAL}
    X = encoder(X_brut, encodeurs)
    idx = np.arange(len(df))
    i_tr, i_vt = train_test_split(idx, train_size=0.8, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_vt, train_size=0.5, stratify=y[i_vt], random_state=SEED)
    print(f"Données : {len(df):,} ménages, {len(FEATURE_NAMES)} variables, {y.mean():.1%} éligibles "
          f"| train {len(i_tr):,} / val {len(i_va):,} / test {len(i_te):,}")

    modele = XGBClassifier(**XGB_PARAMS).fit(X[i_tr], y[i_tr])
    artifact = {"modele": modele, "encodeurs": encodeurs, "seuil": SEUIL,
                "metadata": {"version": "", "zone_grise": {"probabilite": list(ZONE_GRISE)}}}
    em = EligibilityModel(artifact)

    # comparaison sur la validation : régression logistique, grille Excel (si présente), XGBoost
    sc = RobustScaler().fit(X[i_tr])
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(X[i_tr]), y[i_tr])
    comparaison = [{"modele": "regression_logistique", **mesures(y[i_va], lr.predict(sc.transform(X[i_va])))}]
    if "decision_grille_excel" in df:
        comparaison.append({"modele": "grille_excel", **mesures(y[i_va], df["decision_grille_excel"].to_numpy()[i_va])})
    p_va = modele.predict_proba(X[i_va])[:, 1]
    comparaison.append({"modele": "xgboost", **mesures(y[i_va], (p_va >= SEUIL).astype(int), p_va)})
    for r in comparaison:
        print(f"  validation {r['modele']:22s} accuracy={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"précision={r['precision']:.4f} rappel={r['rappel']:.4f}")

    p_te = modele.predict_proba(X[i_te])[:, 1]
    pred_te = (p_te >= SEUIL).astype(int)
    test = {"seuil": SEUIL, **mesures(y[i_te], pred_te, p_te)}
    print(f"Test : {json.dumps(test, ensure_ascii=False)}")

    validation_croisee = None
    if not args.sans_cv:
        cv = cross_validate(XGBClassifier(**XGB_PARAMS), X[i_tr], y[i_tr], cv=5,
                            scoring={"accuracy": "accuracy", "f1": "f1", "rappel": "recall", "precision": "precision"})
        validation_croisee = {m: {"moyenne": round(float(cv[f"test_{m}"].mean()), 4), "min": round(float(cv[f"test_{m}"].min()), 4)}
                              for m in ["accuracy", "f1", "rappel", "precision"]}
        print(f"Validation croisée (5 parties) : {json.dumps(validation_croisee, ensure_ascii=False)}")

    erreurs = pred_te != y[i_te]
    a_verifier = em.a_verifier(p_te)
    zone_grise = {"probabilite": list(ZONE_GRISE), "part_menages": round(float(a_verifier.mean()), 4),
                  "part_erreurs": round(float((a_verifier & erreurs).sum() / max(erreurs.sum(), 1)), 4)}

    grille = None
    if "decision_grille_excel" in df:
        g = df["decision_grille_excel"].to_numpy()[i_te]
        oublies, a_tort = (y[i_te] == 1) & (g == 0), (y[i_te] == 0) & (g == 1)
        grille = {**mesures(y[i_te], g), "eligibles_oublies": int(oublies.sum()),
                  "oublies_retrouves_par_le_modele": round(float(pred_te[oublies].mean()), 4),
                  "retenus_a_tort": int(a_tort.sum()), "a_tort_ecartes_par_le_modele": round(float(1 - pred_te[a_tort].mean()), 4)}

    assistance = None
    if "assistance_ideale" in df:
        artifact["modele_assistance"], artifact["classes_assistance"], assistance = entrainer_assistance(
            df, X_brut, encodeurs, y, cv=not args.sans_cv)

    contrib, _ = em.contributions(X_brut.iloc[i_te[:5000]])
    moy = np.abs(contrib).mean(0)
    importance = sorted([{"feature": f, "theme": THEMES[f], "importance": round(float(v), 4)} for f, v in zip(FEATURE_NAMES, moy)],
                        key=lambda r: r["importance"], reverse=True)
    themes = pd.Series(moy, index=[THEMES[f] for f in FEATURE_NAMES]).groupby(level=0).sum().sort_values(ascending=False)

    metadata = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "modele": "XGBoost",
        "parametres": {k: v for k, v in XGB_PARAMS.items() if k != "random_state"},
        "seuil": SEUIL, "taux_eligibles_entrainement": round(float(y.mean()), 4), "n_entrainement": len(i_tr),
        "n_variables": len(FEATURE_NAMES), "donnees": str(args.data.name),
        "cible": "décision de la vérification approfondie (la grille Excel n'est pas utilisée)",
        "validation_comparaison": comparaison, "test": test, "validation_croisee": validation_croisee,
        "zone_grise": zone_grise, "grille_excel_test": grille, "assistance": assistance,
        "priorite": {"methode": "log_odds"},
        "importance_globale": importance,
        "importance_themes": [{"theme": t, "importance": round(float(v), 4)} for t, v in themes.items()],
        "variables": FEATURE_NAMES,
    }
    artifact["metadata"] = metadata
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out / "model.joblib", compress=3)  # environ 20 Mo au lieu de 60
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artefacts écrits dans {args.out}")
    return metadata


if __name__ == "__main__":
    main()
