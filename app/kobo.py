"""Lecture des soumissions KoboToolbox (webhook REST Service et API v2)."""
from __future__ import annotations

from datetime import datetime

import httpx

from .config import get_settings
from .schema import IDENTITY_FIELDS


def flatten(payload: dict) -> dict:
    """Kobo préfixe les questions par leurs groupes ('menage/taille_menage').

    On garde le dernier segment, ce qui rend l'API insensible à la façon dont
    les questions sont regroupées dans le formulaire.
    """
    flat = {}
    for key, value in payload.items():
        if key.startswith("_") or isinstance(value, (list, dict)):
            continue
        flat[key.rsplit("/", 1)[-1]] = value
    return flat


def consent_refused(payload: dict) -> bool:
    """Le formulaire s'arrête si le ménage refuse : rien à évaluer ni à stocker."""
    return flatten(payload).get("consentement") == "non"


def submission_key(payload: dict) -> tuple[str, str | None]:
    """Identifiant stable de la soumission : _id Kobo, sinon _uuid / instanceID."""
    uuid = payload.get("_uuid") or str(payload.get("meta/instanceID", "")).removeprefix("uuid:") or None
    if payload.get("_id") is not None:
        return f"kobo-{payload['_id']}", uuid
    if uuid:
        return f"uuid-{uuid}", uuid
    raise ValueError("Soumission sans _id ni _uuid : impossible de l'identifier.")


def submission_time(payload: dict) -> datetime | None:
    raw = payload.get("_submission_time") or payload.get("end")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def identity(flat: dict) -> dict:
    return {k: (str(flat[k])[:200] if flat.get(k) not in (None, "") else None) for k in IDENTITY_FIELDS}


def fetch_submissions() -> list[dict]:
    """Récupère toutes les soumissions du formulaire via l'API Kobo v2 (rattrapage)."""
    s = get_settings()
    if not (s.kobo_api_token and s.kobo_asset_uid):
        raise RuntimeError("KOBO_API_TOKEN et KOBO_ASSET_UID doivent être configurés pour la synchronisation.")
    url = f"{s.kobo_server_url}/api/v2/assets/{s.kobo_asset_uid}/data.json?limit=1000"
    headers = {"Authorization": f"Token {s.kobo_api_token}"}
    results = []
    with httpx.Client(timeout=60) as client:
        while url:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            results.extend(data.get("results", []))
            url = data.get("next")
    return results
