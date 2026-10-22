"""Modèle de données : ménages évalués et historique des revues humaines."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Beneficiary(Base):
    __tablename__ = "beneficiaires"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kobo_key: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    kobo_uuid: Mapped[str | None] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(20))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    nom_chef_menage: Mapped[str | None] = mapped_column(String(200))
    telephone: Mapped[str | None] = mapped_column(String(50))
    village: Mapped[str | None] = mapped_column(String(200))
    enumerateur: Mapped[str | None] = mapped_column(String(200))
    gps: Mapped[str | None] = mapped_column(String(200))

    features: Mapped[dict] = mapped_column(JSON)
    raw_payload: Mapped[dict] = mapped_column(JSON)

    score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_decision: Mapped[bool] = mapped_column(Boolean)
    a_verifier: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    base_value: Mapped[float] = mapped_column(Float)
    explanation: Mapped[list] = mapped_column(JSON)
    summary: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(50))

    # décision finale = décision du modèle, sauf si un humain l'a désapprouvée
    final_decision: Mapped[bool] = mapped_column(Boolean, index=True)
    statut_revue: Mapped[str] = mapped_column(String(20), default="en_attente", index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    reviews: Mapped[list["Review"]] = relationship(
        back_populates="beneficiary", order_by="Review.created_at.desc()", cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "revues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    beneficiary_id: Mapped[int] = mapped_column(ForeignKey("beneficiaires.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(20))  # "approuve" | "desapprouve" | "reinitialise"
    decision_avant: Mapped[bool] = mapped_column(Boolean)
    decision_apres: Mapped[bool] = mapped_column(Boolean)
    commentaire: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    beneficiary: Mapped[Beneficiary] = relationship(back_populates="reviews")


class Selection(Base):
    """Liste de ménages retenus quand le budget ne couvre pas tous les éligibles (instantané daté)."""
    __tablename__ = "selections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nom: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[str] = mapped_column(String(100))
    nb_demande: Mapped[int] = mapped_column(Integer)
    nb_eligibles: Mapped[int] = mapped_column(Integer)
    nb_retenus: Mapped[int] = mapped_column(Integer)
    repartition: Mapped[str] = mapped_column(String(30))  # "globale" | "proportionnelle_zone"
    inclure_non_revus: Mapped[bool] = mapped_column(Boolean)
    model_version: Mapped[str] = mapped_column(String(50))
    methode_priorite: Mapped[str] = mapped_column(String(20))  # "score_predit" | "log_odds"

    items: Mapped[list["SelectionItem"]] = relationship(
        back_populates="selection", order_by="SelectionItem.rang", cascade="all, delete-orphan")


class SelectionItem(Base):
    """Un ménage éligible classé dans une sélection, retenu ou non."""
    __tablename__ = "selection_menages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    selection_id: Mapped[int] = mapped_column(ForeignKey("selections.id"), index=True)
    beneficiary_id: Mapped[int] = mapped_column(ForeignKey("beneficiaires.id"))
    rang: Mapped[int] = mapped_column(Integer)  # rang de priorité dans sa zone (proportionnelle) ou global
    priorite: Mapped[float] = mapped_column(Float)
    zone: Mapped[str] = mapped_column(String(100))
    retenu: Mapped[bool] = mapped_column(Boolean)

    selection: Mapped[Selection] = relationship(back_populates="items")
    beneficiary: Mapped[Beneficiary] = relationship()


_engine = None
SessionLocal = sessionmaker(expire_on_commit=False)


def init_db(url: str | None = None):
    global _engine
    url = url or get_settings().database_url
    kwargs = {}
    if url.startswith("sqlite"):
        if "///" in url and ":memory:" not in url:
            Path(url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
    _engine = create_engine(url, **kwargs)
    Base.metadata.create_all(_engine)
    SessionLocal.configure(bind=_engine)
    return _engine


def get_session():
    with SessionLocal() as session:
        yield session
