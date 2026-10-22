"""Envoie des ménages simulés au webhook, au format KoboToolbox (questions du formulaire v5).

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
sys.path.insert(0, str(ROOT / "notebooks"))

import generateur_ciblage_v5 as gen  # noqa: E402
from app.schema import FEATURES  # noqa: E402
from kobo.build_xlsform import GROUPS  # noqa: E402

GROUPE = {name: groupe for groupe, _, names in GROUPS for name in names}
PRENOMS = ["Amani", "Grace", "Esther", "Jean", "Furaha", "Neema", "Baraka", "Josué", "Aline", "Patient",
           "Sifa", "Riziki", "Daniel", "Mapendo", "Christian", "Zawadi", "Faida", "Moïse"]
NOMS = ["Kahindo", "Muhindo", "Kavira", "Paluku", "Masika", "Kambale", "Mumbere", "Kasereka", "Bahati", "Ngabo"]


def to_payload(row: pd.Series, i: int) -> dict:
    u = str(uuid.uuid4())
    when = datetime.now(timezone.utc) - timedelta(minutes=random.randint(5, 60 * 24 * 6))
    p = {
        "_id": 100000 + i, "_uuid": u, "_submission_time": when.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta/instanceID": f"uuid:{u}", "consentement": "oui",
        "identification/enumerateur": random.choice(["Grace M.", "Patient K.", "Aline B."]),
        "identification/village": f"{row['localite']} - quartier {random.randint(1, 6)}",
        "identification/nom_chef_menage": f"{random.choice(PRENOMS)} {random.choice(NOMS)}",
        "identification/telephone": f"+243 99{random.randint(1000000, 9999999)}",
    }
    for f in FEATURES:
        valeur = row[f.name]
        if pd.isna(valeur):  # question non posée (logique de saut) : absente de la soumission, comme dans Kobo
            continue
        p[f"{GROUPE[f.name]}/{f.name}"] = str(int(valeur)) if f.kind == "integer" else str(valeur)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    menages = gen.generer(max(args.n * 3, 300), graine=args.seed)
    # sur-représente les éligibles pour une démo plus parlante
    k = min(args.n // 2, int(menages.eligible.sum()))
    echantillon = pd.concat([menages[menages.eligible == 1].sample(k, random_state=args.seed),
                             menages[menages.eligible == 0].sample(args.n - k, random_state=args.seed)])
    ok = 0
    with httpx.Client(timeout=30) as client:
        for i, (_, row) in enumerate(echantillon.sample(frac=1, random_state=args.seed).iterrows()):
            r = client.post(f"{args.url.rstrip('/')}/api/kobo/webhook", params={"token": args.token},
                            json=to_payload(row, i))
            ok += r.status_code == 200
            if r.status_code != 200:
                print(r.status_code, r.text)
    print(f"{ok}/{len(echantillon)} soumissions acceptées")


if __name__ == "__main__":
    main()
