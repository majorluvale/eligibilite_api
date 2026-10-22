"""Envoie des ménages du jeu de données au webhook, au format KoboToolbox.

Utile pour une démonstration sans enquête terrain :
    python scripts/simulate_kobo.py --url http://localhost:8000 --token <KOBO_WEBHOOK_TOKEN> -n 40
"""
from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schema import FEATURES  # noqa: E402

GROUP = {"location": "menage", "statut_menage": "menage", "sexe_chef_menage": "menage",
         "age_chef_menage": "menage", "taille_menage": "menage", "membre_handicap": "menage",
         "femme_enceinte_allaitante": "menage", "type_logement": "conditions_vie",
         "source_eau": "conditions_vie", "source_revenu": "economie", "nombre_repas_jour": "economie",
         "dette_contractee_crise": "economie"}
PRENOMS = ["Amani", "Grace", "Esther", "Jean", "Furaha", "Neema", "Baraka", "Josué", "Aline", "Patient",
           "Sifa", "Riziki", "Daniel", "Mapendo", "Christian", "Zawadi", "Faida", "Moïse"]
NOMS = ["Kahindo", "Muhindo", "Kavira", "Paluku", "Masika", "Kambale", "Mumbere", "Kasereka", "Bahati", "Ngabo"]
VILLAGES = ["Rutshuru", "Rwanguba", "Binza", "Bambo", "Kibirizi", "Birambizo"]


def to_payload(row: pd.Series, i: int) -> dict:
    sid = 100000 + i
    u = str(uuid.uuid4())
    when = datetime.now(timezone.utc) - timedelta(minutes=random.randint(5, 60 * 24 * 6))
    p = {
        "_id": sid, "_uuid": u, "_submission_time": when.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta/instanceID": f"uuid:{u}", "consentement": "oui",
        "identification/enumerateur": random.choice(["Grace M.", "Patient K.", "Aline B."]),
        "identification/village": random.choice(VILLAGES),
        "identification/nom_chef_menage": f"{random.choice(PRENOMS)} {random.choice(NOMS)}",
        "identification/telephone": f"+243 99{random.randint(1000000, 9999999)}",
    }
    for f in FEATURES:
        value = row[f.name]
        if f.kind == "categorical":
            value = {v: k for k, v in f.choices.items()}[value]
        p[f"{GROUP[f.kobo]}/{f.kobo}"] = str(value)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--data", default=str(ROOT.parent / "dataset_humanitaire_complexe_100k.csv"))
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    # sur-représente les éligibles pour une démo plus parlante
    sample = pd.concat([df[df.Eligible == 1].sample(args.n // 2, random_state=args.seed),
                        df[df.Eligible == 0].sample(args.n - args.n // 2, random_state=args.seed)])
    ok = 0
    with httpx.Client(timeout=30) as client:
        for i, (_, row) in enumerate(sample.sample(frac=1, random_state=args.seed).iterrows()):
            r = client.post(f"{args.url.rstrip('/')}/api/kobo/webhook", params={"token": args.token},
                            json=to_payload(row, i))
            ok += r.status_code == 200
            if r.status_code != 200:
                print(r.status_code, r.text)
    print(f"{ok}/{len(sample)} soumissions acceptées")


if __name__ == "__main__":
    main()
