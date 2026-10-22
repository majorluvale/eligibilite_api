"""Configuration lue depuis les variables d'environnement (voir .env.example)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("eligibilite")


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    database_url: str
    dashboard_users: dict[str, str]
    secret_key: str
    kobo_webhook_token: str
    kobo_server_url: str
    kobo_api_token: str
    kobo_asset_uid: str
    review_band: float | None  # None = marge enregistrée avec le modèle
    model_dir: str | None


@lru_cache
def get_settings() -> Settings:
    _load_dotenv()
    db = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'eligibilite.db'}")
    # Render / Neon / Supabase fournissent "postgres://", SQLAlchemy veut un driver explicite
    if db.startswith("postgres://"):
        db = "postgresql+psycopg://" + db[len("postgres://"):]
    elif db.startswith("postgresql://"):
        db = "postgresql+psycopg://" + db[len("postgresql://"):]

    users = {}
    for pair in os.getenv("DASHBOARD_USERS", "admin:admin").split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            users[u.strip()] = p.strip()
    if users.get("admin") == "admin":
        log.warning("DASHBOARD_USERS utilise le compte par défaut admin:admin — à changer en production.")

    token = os.getenv("KOBO_WEBHOOK_TOKEN", "")
    if not token:
        log.warning("KOBO_WEBHOOK_TOKEN vide : le webhook Kobo accepte toute requête.")

    return Settings(
        database_url=db,
        dashboard_users=users,
        secret_key=os.getenv("SECRET_KEY", "dev-secret-a-changer"),
        kobo_webhook_token=token,
        kobo_server_url=os.getenv("KOBO_SERVER_URL", "https://kf.kobotoolbox.org").rstrip("/"),
        kobo_api_token=os.getenv("KOBO_API_TOKEN", ""),
        kobo_asset_uid=os.getenv("KOBO_ASSET_UID", ""),
        review_band=float(os.environ["REVIEW_BAND"]) if os.getenv("REVIEW_BAND") else None,
        model_dir=os.getenv("MODEL_DIR") or None,
    )
