"""Source unique de vérité pour les questions du formulaire.

Chaque variable du modèle est décrite une seule fois ici : son nom dans les
données d'entraînement, son nom dans le formulaire Kobo, son libellé et ses
choix (code Kobo -> valeur d'entraînement). Le générateur XLSForm
(kobo/build_xlsform.py), l'API et le modèle utilisent tous ce fichier, ce qui
garantit que le formulaire Kobo reste compatible avec le modèle.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str                     # colonne dans le jeu d'entraînement
    kobo: str                     # nom de la question dans Kobo
    label: str                    # libellé affiché (formulaire + dashboard)
    kind: str                     # "categorical" | "integer"
    choices: dict[str, str] = field(default_factory=dict)  # code Kobo -> valeur d'entraînement
    min: int | None = None
    max: int | None = None
    hint: str = ""


OUI_NON = {"oui": "Oui", "non": "Non"}

FEATURES: list[Feature] = [
    Feature("Location", "location", "Zone / localité du ménage", "categorical", {
        "zone_a": "Zone A (Stable)",
        "zone_b": "Zone B (Rural)",
        "zone_c": "Zone C (Rural)",
        "zone_d": "Zone D (Instable)",
        "zone_e": "Zone E (Crise)",
        "zone_f": "Zone F (Urgence)",
    }),
    Feature("Statut_Menage", "statut_menage", "Statut du ménage", "categorical", {
        "hote": "Non Déplacé (Hôte)",
        "idp": "Déplacé Interne (IDP)",
        "retourne": "Retourné",
        "refugie": "Réfugié",
    }),
    Feature("Sexe_Chef_Menage", "sexe_chef_menage", "Sexe du chef de ménage", "categorical", {
        "m": "M",
        "f": "F",
    }),
    Feature("Age_Chef_Menage", "age_chef_menage", "Âge du chef de ménage (années)", "integer",
            min=10, max=110),
    Feature("Taille_Menage", "taille_menage", "Nombre de personnes dans le ménage", "integer",
            min=1, max=80, hint="Toutes les personnes qui vivent et mangent ensemble"),
    Feature("Membre_Handicap_Maladie_Chronique", "membre_handicap",
            "Le ménage a-t-il un membre vivant avec un handicap ou une maladie chronique ?",
            "categorical", OUI_NON),
    Feature("Femme_Enceinte_Allaitante", "femme_enceinte_allaitante",
            "Le ménage a-t-il une femme enceinte ou allaitante ?", "categorical", OUI_NON),
    Feature("Type_Logement", "type_logement", "Type de logement", "categorical", {
        "dur": "Maison en dur (Briques/Ciment)",
        "banco": "Maison en terre (Banco)",
        "transitionnel": "Abri transitionnel (Bois/Bâche)",
        "urgence": "Abri d’urgence (Paille/Tente)",
    }),
    Feature("Source_Eau", "source_eau", "Principale source d'eau de boisson", "categorical", {
        "robinet": "Robinet privé",
        "borne": "Borne fontaine publique",
        "puits_protege": "Puits protégé",
        "puits_ouvert": "Puits ouvert/Non protégé",
        "surface": "Eau de surface (Rivière/Marigot)",
    }),
    Feature("Source_Revenu", "source_revenu", "Principale source de revenu", "categorical", {
        "salarie": "Salarié régulier",
        "commercant": "Commerçant",
        "agriculteur": "Agriculteur/Eleveur",
        "journalier": "Travailleur journalier",
        "sans_emploi": "Sans emploi/Mendicité",
    }),
    Feature("Nombre_Repas_Jour", "nombre_repas_jour",
            "Nombre de repas pris par jour par les adultes (7 derniers jours)", "integer",
            min=1, max=3, hint="Choisir 1 si moins d'un repas par jour, 3 si trois ou plus"),
    Feature("Dette_Contractee_Crise", "dette_contractee_crise",
            "Le ménage a-t-il contracté une dette depuis le début de la crise ?",
            "categorical", OUI_NON),
]

# Libellés courts pour les graphiques et tableaux du dashboard.
SHORT_LABELS = {
    "Location": "Zone",
    "Statut_Menage": "Statut du ménage",
    "Sexe_Chef_Menage": "Sexe du chef de ménage",
    "Age_Chef_Menage": "Âge du chef de ménage",
    "Taille_Menage": "Taille du ménage",
    "Membre_Handicap_Maladie_Chronique": "Handicap / maladie chronique",
    "Femme_Enceinte_Allaitante": "Femme enceinte / allaitante",
    "Type_Logement": "Type de logement",
    "Source_Eau": "Source d'eau",
    "Source_Revenu": "Source de revenu",
    "Nombre_Repas_Jour": "Repas par jour",
    "Dette_Contractee_Crise": "Dette depuis la crise",
}

FEATURES_BY_NAME = {f.name: f for f in FEATURES}
FEATURES_BY_KOBO = {f.kobo: f for f in FEATURES}
FEATURE_NAMES = [f.name for f in FEATURES]
CATEGORICAL = [f.name for f in FEATURES if f.kind == "categorical"]
NUMERIC = [f.name for f in FEATURES if f.kind == "integer"]

# Questions d'identification : stockées et affichées, jamais utilisées par le modèle.
IDENTITY_FIELDS = {
    "nom_chef_menage": "Nom du chef de ménage",
    "telephone": "Téléphone",
    "village": "Village / quartier",
    "enumerateur": "Enquêteur",
    "gps": "Coordonnées GPS",
}


class ValidationError(ValueError):
    pass


def normalize_features(values: dict) -> dict:
    """Convertit des réponses (codes Kobo OU valeurs d'entraînement) en valeurs du modèle.

    Accepte pour chaque variable soit son nom Kobo soit son nom d'entraînement.
    Lève ValidationError avec la liste de tous les problèmes trouvés.
    """
    out, errors = {}, []
    for f in FEATURES:
        raw = values.get(f.kobo, values.get(f.name))
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors.append(f"{f.kobo}: valeur manquante")
            continue
        if f.kind == "integer":
            try:
                v = int(float(raw))
            except (TypeError, ValueError):
                errors.append(f"{f.kobo}: entier attendu, reçu {raw!r}")
                continue
            if (f.min is not None and v < f.min) or (f.max is not None and v > f.max):
                errors.append(f"{f.kobo}: {v} hors de l'intervalle [{f.min}, {f.max}]")
                continue
            out[f.name] = v
        else:
            raw = str(raw).strip()
            if raw in f.choices:
                out[f.name] = f.choices[raw]
            elif raw in f.choices.values():
                out[f.name] = raw
            else:
                errors.append(f"{f.kobo}: choix inconnu {raw!r} (attendu: {', '.join(f.choices)})")
    if errors:
        raise ValidationError("; ".join(errors))
    return out
