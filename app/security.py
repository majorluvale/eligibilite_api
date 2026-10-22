"""Authentification du dashboard (HTTP Basic), jeton CSRF et jeton du webhook Kobo."""
from __future__ import annotations

import hashlib
import hmac
import secrets

from fastapi import Depends, Header, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import get_settings

basic = HTTPBasic(realm="Dashboard eligibilite")


def current_user(credentials: HTTPBasicCredentials = Depends(basic)) -> str:
    users = get_settings().dashboard_users
    expected = users.get(credentials.username, "")
    ok = secrets.compare_digest(credentials.password.encode(), expected.encode()) and expected != ""
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Identifiants invalides",
                            headers={"WWW-Authenticate": 'Basic realm="Dashboard eligibilite"'})
    return credentials.username


def csrf_token(username: str) -> str:
    key = get_settings().secret_key.encode()
    return hmac.new(key, f"csrf:{username}".encode(), hashlib.sha256).hexdigest()


def check_csrf(username: str, token: str) -> None:
    if not secrets.compare_digest(token or "", csrf_token(username)):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Jeton CSRF invalide")


def check_webhook_token(
    token: str | None = Query(default=None),
    x_webhook_token: str | None = Header(default=None),
) -> None:
    expected = get_settings().kobo_webhook_token
    if not expected:
        return
    given = x_webhook_token or token or ""
    if not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Jeton webhook invalide")
