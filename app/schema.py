"""Source unique de vérité pour les questions du formulaire.

Chaque question est décrite une seule fois ici : son nom (identique dans Kobo,
dans les données d'entraînement et dans l'API), son libellé, ses choix
(code -> libellé affiché) et, le cas échéant, sa condition d'affichage.
Le générateur XLSForm (kobo/build_xlsform.py), l'API et le modèle utilisent
tous ce fichier : le formulaire Kobo reste donc toujours compatible avec le modèle.

Les questions reprennent la grille de ciblage de l'ONG (voir
notebooks/generateur_ciblage_v5.py) : statut, profil du chef de ménage, taille,
vulnérabilités, logement, articles essentiels, alimentation, revenu, bétail et
assistance récente.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str                     # nom de la question (Kobo = colonne d'entraînement = clé de l'API)
    label: str                    # libellé de la question dans le formulaire
    short: str                    # libellé court (dashboard, explications)
    kind: str                     # "categorical" | "integer"
    choices: dict[str, str] = field(default_factory=dict)  # code -> libellé affiché
    min: int | None = None
    max: int | None = None
    hint: str = ""
    relevant: tuple[str, frozenset] | None = None  # posée seulement si <question> a l'une de ces valeurs
    modele: bool = True           # False : collectée et affichée, mais jamais donnée au modèle

    @property
    def kobo(self) -> str:
        return self.name


OUI_NON = {"oui": "Oui", "non": "Non"}

FEATURES: list[Feature] = [
    Feature("localite", "Localité du ménage", "Localité", "categorical", {
        "Rutshuru": "Rutshuru", "Rwanguba": "Rwanguba", "Binza": "Binza",
        "Bambo": "Bambo", "Kibirizi": "Kibirizi", "Birambizo": "Birambizo",
    }, hint="Sert à répartir le budget par zone ; n'influence pas la décision", modele=False),
    Feature("statut_menage", "Statut du ménage", "Statut du ménage", "categorical", {
        "resident": "Résident", "famille_accueil": "Famille d'accueil",
        "deplace": "Déplacé", "retourne": "Retourné",
    }),
    Feature("duree_deplacement_mois", "Depuis combien de mois le ménage est-il déplacé ?", "Durée du déplacement (mois)",
            "integer", min=0, max=120, hint="0 si moins d'un mois",
            relevant=("statut_menage", frozenset({"deplace"}))),
    Feature("sexe_chef_menage", "Sexe du chef de ménage", "Sexe du chef de ménage", "categorical",
            {"homme": "Homme", "femme": "Femme"}),
    Feature("age_chef_menage", "Âge du chef de ménage (années)", "Âge du chef de ménage", "integer", min=10, max=110),
    Feature("taille_menage", "Nombre de personnes dans le ménage", "Taille du ménage", "integer", min=1, max=40,
            hint="Toutes les personnes qui vivent et mangent ensemble, chef de ménage compris"),
    Feature("enfants_moins_5", "Nombre d'enfants de moins de 5 ans dans le ménage", "Enfants de moins de 5 ans",
            "integer", min=0, max=20),
    Feature("membre_handicap", "Un membre du ménage vit-il avec un handicap ?", "Membre handicapé", "categorical", OUI_NON),
    Feature("membre_maladie_chronique", "Un membre du ménage a-t-il une maladie chronique ?", "Maladie chronique",
            "categorical", OUI_NON),
    Feature("femme_enceinte_allaitante", "Le ménage a-t-il une femme enceinte ou allaitante ?", "Femme enceinte / allaitante",
            "categorical", OUI_NON),
    Feature("enfant_non_accompagne_heberge", "Le ménage héberge-t-il un enfant non accompagné ou séparé ?",
            "Enfant non accompagné hébergé", "categorical", OUI_NON),
    Feature("type_logement", "Type de logement actuel", "Logement", "categorical", {
        "maison_correcte": "Maison en bon état", "maison_endommagee": "Maison endommagée",
        "heberge_famille": "Hébergé par une famille", "abri_fortune": "Abri de fortune", "sans_abri": "Sans abri",
    }),
    Feature("articles_essentiels", "Combien de ces 5 articles le ménage possède-t-il en quantité suffisante ?",
            "Articles essentiels (sur 5)", "integer", min=0, max=5,
            hint="Couvertures, nattes, bidons, ustensiles de cuisine, moustiquaires"),
    Feature("repas_par_jour", "Nombre de repas pris par jour par les adultes (7 derniers jours)", "Repas par jour",
            "integer", min=1, max=3),
    Feature("jours_sans_nourriture_7j", "Au cours des 7 derniers jours, combien de jours le ménage n'a-t-il rien eu à manger ?",
            "Jours sans nourriture (7 j)", "integer", min=0, max=7),
    Feature("source_revenu", "Principale source de revenu du ménage", "Source de revenu", "categorical", {
        "salaire_regulier": "Salaire régulier", "petit_commerce": "Petit commerce", "agriculture": "Agriculture",
        "travail_journalier": "Travail journalier", "aucun_revenu": "Aucun revenu / mendicité",
    }),
    Feature("possede_betail", "Le ménage possède-t-il du bétail (au moins 2 chèvres ou équivalent) ?", "Possède du bétail",
            "categorical", OUI_NON),
    Feature("assistance_3_derniers_mois", "Le ménage a-t-il reçu une assistance humanitaire au cours des 3 derniers mois ?",
            "Assistance reçue (3 mois)", "categorical", OUI_NON),
    Feature("type_assistance_recue", "Quel type d'assistance ?", "Type d'assistance reçue", "categorical", {
        "vivres": "Vivres", "cash": "Cash", "articles_menagers": "Articles ménagers", "abri": "Abri", "autre": "Autre",
    }, relevant=("assistance_3_derniers_mois", frozenset({"oui"}))),
]

FEATURES_BY_NAME = {f.name: f for f in FEATURES}
FEATURES_BY_KOBO = FEATURES_BY_NAME
FEATURE_NAMES = [f.name for f in FEATURES if f.modele]      # variables d'entrée du modèle
CATEGORICAL = [n for n in FEATURE_NAMES if FEATURES_BY_NAME[n].kind == "categorical"]
NUMERIC = [n for n in FEATURE_NAMES if FEATURES_BY_NAME[n].kind == "integer"]
SHORT_LABELS = {f.name: f.short for f in FEATURES}
ZONE_FIELD = "localite"  # question utilisée pour la répartition du budget par zone

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


def libelle(name: str, value) -> str:
    """Libellé affiché d'une réponse (code -> texte), pour le dashboard et les explications."""
    f = FEATURES_BY_NAME.get(name)
    if value is None:
        return "non applicable"
    if f and f.choices:
        return f.choices.get(str(value), str(value))
    return str(value)


def _est_posee(f: Feature, valeurs: dict) -> bool:
    if f.relevant is None:
        return True
    question, attendues = f.relevant
    return valeurs.get(question) in attendues


def normalize_features(values: dict) -> dict:
    """Convertit les réponses Kobo (codes ou libellés) en valeurs du modèle.

    Une question conditionnelle non posée vaut None. Lève ValidationError avec
    la liste de tous les problèmes trouvés.
    """
    out, errors = {}, []
    for f in FEATURES:
        if not _est_posee(f, out):  # les conditions portent sur des questions déjà lues
            out[f.name] = None
            continue
        raw = values.get(f.name)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors.append(f"{f.name}: valeur manquante")
            continue
        if f.kind == "integer":
            try:
                v = int(float(raw))
            except (TypeError, ValueError):
                errors.append(f"{f.name}: entier attendu, reçu {raw!r}")
                continue
            if (f.min is not None and v < f.min) or (f.max is not None and v > f.max):
                errors.append(f"{f.name}: {v} hors de l'intervalle [{f.min}, {f.max}]")
                continue
            out[f.name] = v
        else:
            raw = str(raw).strip()
            if raw in f.choices:
                out[f.name] = raw
            else:
                code = next((c for c, lib in f.choices.items() if lib == raw), None)
                if code is None:
                    errors.append(f"{f.name}: choix inconnu {raw!r} (attendu : {', '.join(f.choices)})")
                    continue
                out[f.name] = code
    if not errors and out["enfants_moins_5"] >= out["taille_menage"]:
        errors.append("enfants_moins_5: doit être inférieur à la taille du ménage (le chef de ménage en fait partie)")
    if errors:
        raise ValidationError("; ".join(errors))
    return out
