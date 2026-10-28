"""Envoie de vraies soumissions dans le projet KoboToolbox déjà déployé (API OpenRosa).

Contrairement à scripts/simulate_kobo.py (qui envoie au webhook de l'API), ce script
soumet directement au serveur KoboToolbox, comme le ferait KoboCollect ou Enketo :
les soumissions apparaissent dans le projet Kobo, pas seulement dans notre base.

La structure exacte attendue (nom de la racine, version, `formhub/uuid`, `__version__`)
est lue en direct sur le serveur à chaque exécution : KoboCAT renomme la racine de
l'instance avec son identifiant interne, différent du nom défini dans schema.py, et ce
renommage ne peut pas être deviné à l'avance. Le script reste donc valide après un
redéploiement du formulaire.

Aucune donnée GPS n'est inventée : le champ `gps` est envoyé vide, ce que le formulaire
autorise (il n'est pas obligatoire).

Usage :
    python scripts/submit_to_kobo.py --dry-run -n 3     # affiche les XML, n'envoie rien
    python scripts/submit_to_kobo.py -n 100              # envoie, avec un essai de 3 d'abord
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import uuid
import xml.sax.saxutils as sax
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import generateur_donnees as gen  # noqa: E402
from app.config import get_settings  # noqa: E402
from kobo.build_xlsform import GROUPS  # noqa: E402

GROUPE = {name: groupe for groupe, _, names in GROUPS for name in names}
PRENOMS = ["Amani", "Grace", "Esther", "Jean", "Furaha", "Neema", "Baraka", "Josué", "Aline", "Patient",
           "Sifa", "Riziki", "Daniel", "Mapendo", "Christian", "Zawadi", "Faida", "Moïse"]
NOMS = ["Kahindo", "Muhindo", "Kavira", "Paluku", "Masika", "Kambale", "Mumbere", "Kasereka", "Bahati", "Ngabo"]
ENQUETEURS = ["Grace M.", "Patient K.", "Aline B.", "Daniel S.", "Riziki N."]


def deployed_structure(server: str, asset_uid: str, token: str) -> dict:
    """Lit sur le serveur Kobo la structure exacte attendue par les soumissions.

    KoboCAT renomme la racine de l'instance avec son propre identifiant (différent de
    l'`id_string` du formulaire) et calcule `formhub/uuid` et `__version__` : impossible
    à deviner sans interroger le serveur.
    """
    hdr = {"Authorization": f"Token {token}"}
    with httpx.Client(timeout=30, headers=hdr) as c:
        asset = c.get(f"{server}/api/v2/assets/{asset_uid}/").raise_for_status().json()
        if not asset.get("deployment__active"):
            raise RuntimeError("Le formulaire n'est pas déployé sur KoboToolbox.")
        lien_export = asset["deployment__data_download_links"]["csv_legacy"]
        m = re.match(r"(https://[^/]+)/([^/]+)/exports/([^/]+)/", lien_export)
        if not m:
            raise RuntimeError(f"Lien d'export inattendu : {lien_export}")
        kc_server, username, form_id = m.groups()

        formlist = c.get(f"{kc_server}/{username}/formList").raise_for_status().text
        m = re.search(rf'<xform>(?:(?!</xform>).)*?<formID>{re.escape(form_id)}</formID>'
                      rf'(?:(?!</xform>).)*?<downloadUrl>([^<]+)</downloadUrl>(?:(?!</xform>).)*?</xform>',
                      formlist, re.S)
        if not m:
            raise RuntimeError(f"Formulaire {form_id} absent de {kc_server}/{username}/formList "
                               "(jeton sans accès à ce formulaire ?).")
        xform_xml = c.get(m.group(1)).raise_for_status().text

    m = re.search(r"<instance>\s*<([\w:-]+)([^>]*)>", xform_xml)
    root_tag, attrs = m.group(1), dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
    calc = lambda champ: (re.search(rf'<bind nodeset="/{re.escape(root_tag)}/{champ}"[^>]*calculate="\'([^\']*)\'"',
                                     xform_xml) or [None, ""])[1]
    return {"kc_server": kc_server, "username": username, "root_tag": root_tag,
            "version": attrs.get("version", ""), "formhub_uuid": calc("formhub/uuid"),
            "version_calc": calc("__version__")}


def el(name: str, value=None) -> str:
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return f"<{name}/>"
    return f"<{name}>{sax.escape(str(value))}</{name}>"


def groupe(name: str, contenu: str) -> str:
    return f"<{name}>{contenu}</{name}>"


def build_submission_xml(row: pd.Series, structure: dict, instance_uuid: str, quand: datetime) -> str:
    """XML de soumission pour un ménage, dans la structure exacte lue sur le serveur."""
    champs_par_groupe: dict[str, list[str]] = {}
    for f in gen.FEATURES:
        valeur = row[f.name]
        texte = None if pd.isna(valeur) else (str(int(valeur)) if f.kind == "integer" else str(valeur))
        champs_par_groupe.setdefault(GROUPE[f.name], []).append(el(f.name, texte))

    identification = groupe("identification", "".join([
        el("enumerateur", random.choice(ENQUETEURS)),
        el("village", f"{row['localite']} - quartier {random.randint(1, 6)}"),
        el("nom_chef_menage", f"{random.choice(PRENOMS)} {random.choice(NOMS)}"),
        el("telephone", f"+243 99{random.randint(1000000, 9999999)}"),
        el("gps"),  # jamais rempli : aucune donnée GPS n'est inventée
    ]))
    groupes_schema = "".join(groupe(nom_groupe, "".join(champs_par_groupe[nom_groupe]))
                             for nom_groupe, _, _ in GROUPS)

    corps = "".join([
        f"<formhub>{el('uuid', structure['formhub_uuid'])}</formhub>",
        el("start", quand.strftime("%Y-%m-%dT%H:%M:%S.000Z")),
        el("end", (quand + timedelta(minutes=random.randint(15, 40))).strftime("%Y-%m-%dT%H:%M:%S.000Z")),
        el("today", quand.strftime("%Y-%m-%d")),
        el("deviceid", "simulateur-ciblage"),
        el("intro"), el("consentement", "oui"), el("fin_sans_consentement"),
        identification, groupes_schema, el("fin"),
        el("__version__", structure["version_calc"]),
        groupe("meta", el("instanceID", f"uuid:{instance_uuid}")),
    ])
    racine = structure["root_tag"]
    return (f'<?xml version="1.0" encoding="utf-8"?>'
            f'<{racine} id="{racine}" version="{sax.escape(structure["version"])}">{corps}</{racine}>')


def envoyer(xml_text: str, instance_uuid: str, structure: dict, token: str) -> httpx.Response:
    url = f"{structure['kc_server']}/{structure['username']}/submission"
    files = {"xml_submission_file": (f"{instance_uuid}.xml", xml_text.encode("utf-8"), "text/xml")}
    return httpx.post(url, files=files, headers={"Authorization": f"Token {token}", "X-OpenRosa-Version": "1.0"},
                      timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="affiche les XML sans rien envoyer")
    ap.add_argument("--essai", type=int, default=3, help="nombre de soumissions envoyées avant de continuer")
    args = ap.parse_args()

    s = get_settings()
    if not (s.kobo_api_token and s.kobo_asset_uid):
        sys.exit("KOBO_API_TOKEN et KOBO_ASSET_UID doivent être renseignés dans .env")

    print("Lecture de la structure déployée sur KoboToolbox…")
    structure = deployed_structure(s.kobo_server_url, s.kobo_asset_uid, s.kobo_api_token)
    print(f"  racine={structure['root_tag']} version={structure['version']!r} "
          f"serveur={structure['kc_server']} compte={structure['username']}")

    random.seed(args.seed)
    menages = gen.generer(max(args.n * 2, 300), graine=args.seed)
    k = min(args.n // 2, int(menages.eligible.sum()))  # sur-représente les éligibles, plus parlant pour une démo
    echantillon = pd.concat([menages[menages.eligible == 1].sample(k, random_state=args.seed),
                             menages[menages.eligible == 0].sample(args.n - k, random_state=args.seed)]
                           ).sample(frac=1, random_state=args.seed)

    maintenant = datetime.now(timezone.utc)
    ok = rejets = 0
    for i, (_, row) in enumerate(echantillon.iterrows()):
        instance_uuid = str(uuid.uuid4())
        quand = maintenant - timedelta(minutes=random.randint(5, 60 * 24 * 6))
        xml_text = build_submission_xml(row, structure, instance_uuid, quand)

        if args.dry_run:
            print(xml_text if i < 2 else "...")
            continue

        r = envoyer(xml_text, instance_uuid, structure, s.kobo_api_token)
        if r.status_code in (200, 201):
            ok += 1
        else:
            rejets += 1
            print(f"  [{i + 1}] rejeté ({r.status_code}) : {r.text[:300]}")
        if i + 1 == args.essai and rejets:
            sys.exit(f"Arrêt après l'essai : {rejets}/{args.essai} soumissions rejetées. "
                     "Vérifie le message ci-dessus avant de relancer.")
        if i + 1 == args.essai:
            print(f"Essai ({args.essai}) accepté par KoboToolbox, on continue…")

    if not args.dry_run:
        print(f"{ok}/{len(echantillon)} soumissions acceptées par KoboToolbox"
              + (f", {rejets} rejetées" if rejets else "") + ".")


if __name__ == "__main__":
    main()
