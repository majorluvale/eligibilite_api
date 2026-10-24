"""Source unique de vérité pour les questions du formulaire.

Chaque question est décrite une seule fois ici : son nom (identique dans Kobo,
dans les données d'entraînement et dans l'API), son libellé, ses choix
(code -> libellé affiché), son thème et, le cas échéant, sa condition d'affichage.

Les questions sont celles d'une enquête ménage de type MSNA (plus de 70 variables) :
profil, composition, logement, eau et hygiène, santé, consommation alimentaire,
stratégies d'adaptation, moyens d'existence, protection et assistance reçue.

Le générateur XLSForm (kobo/build_xlsform.py), l'API et le modèle utilisent tous ce
fichier : le formulaire Kobo reste donc toujours compatible avec le modèle.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str                     # nom de la question (Kobo = colonne d'entraînement = clé de l'API)
    label: str                    # libellé de la question dans le formulaire
    short: str                    # libellé court (dashboard, explications)
    kind: str                     # "categorical" | "integer"
    theme: str                    # regroupement dans le formulaire et dans les explications
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
DEPLACE = ("statut_menage", frozenset({"deplace"}))
JOURS = dict(kind="integer", min=0, max=7)


def jours(name, aliment, short):
    return Feature(name, f"Au cours des 7 derniers jours, combien de jours le ménage a-t-il mangé : {aliment} ?",
                   short, theme="Alimentation", hint="0 à 7 jours", **JOURS)


def rcsi(name, strategie, short):
    return Feature(name, f"Au cours des 7 derniers jours, combien de jours le ménage a-t-il dû {strategie} ?",
                   short, theme="Stratégies d'adaptation", hint="0 à 7 jours", **JOURS)


def lcs(name, strategie, short):
    return Feature(name, f"Au cours des 30 derniers jours, le ménage a-t-il dû {strategie} faute de nourriture ou d'argent ?",
                   short, "categorical", "Stratégies d'adaptation", OUI_NON)


FEATURES: list[Feature] = [
    # ------------------------------------------------------------------ profil
    Feature("localite", "Localité du ménage", "Localité", "categorical", "Profil", {
        "Rutshuru": "Rutshuru", "Rwanguba": "Rwanguba", "Binza": "Binza",
        "Bambo": "Bambo", "Kibirizi": "Kibirizi", "Birambizo": "Birambizo",
    }, hint="Sert à répartir le budget par zone ; n'influence pas la décision", modele=False),
    Feature("statut_menage", "Statut du ménage", "Statut du ménage", "categorical", "Profil", {
        "resident": "Résident", "famille_accueil": "Famille d'accueil",
        "deplace": "Déplacé", "retourne": "Retourné",
    }),
    Feature("duree_deplacement_mois", "Depuis combien de mois le ménage est-il déplacé ?", "Durée du déplacement (mois)",
            "integer", "Profil", min=0, max=120, hint="0 si moins d'un mois", relevant=DEPLACE),
    Feature("nombre_deplacements_12m", "Combien de fois le ménage a-t-il dû fuir au cours des 12 derniers mois ?",
            "Déplacements (12 mois)", "integer", "Profil", min=0, max=15,
            relevant=("statut_menage", frozenset({"deplace", "retourne"}))),
    Feature("sexe_chef_menage", "Sexe du chef de ménage", "Sexe du chef de ménage", "categorical", "Profil",
            {"homme": "Homme", "femme": "Femme"}),
    Feature("age_chef_menage", "Âge du chef de ménage (années)", "Âge du chef de ménage", "integer", "Profil", min=10, max=110),
    Feature("etat_civil_chef", "État civil du chef de ménage", "État civil du chef", "categorical", "Profil", {
        "marie": "Marié(e) / union libre", "celibataire": "Célibataire", "veuf": "Veuf / veuve",
        "divorce_separe": "Divorcé(e) / séparé(e)",
    }),
    Feature("instruction_chef", "Niveau d'instruction du chef de ménage", "Instruction du chef", "categorical", "Profil", {
        "aucun": "Aucun", "primaire": "Primaire", "secondaire": "Secondaire", "superieur": "Supérieur",
    }),
    # ------------------------------------------------------------------ composition
    Feature("taille_menage", "Nombre de personnes dans le ménage", "Taille du ménage", "integer", "Composition",
            min=1, max=40, hint="Toutes les personnes qui vivent et mangent ensemble, chef de ménage compris"),
    Feature("enfants_moins_5", "Nombre d'enfants de moins de 5 ans", "Enfants de moins de 5 ans", "integer", "Composition",
            min=0, max=20),
    Feature("enfants_5_17", "Nombre d'enfants de 5 à 17 ans", "Enfants de 5 à 17 ans", "integer", "Composition", min=0, max=25),
    Feature("personnes_60_plus", "Nombre de personnes de 60 ans ou plus", "Personnes de 60 ans et plus", "integer",
            "Composition", min=0, max=10),
    Feature("enfants_non_scolarises", "Parmi les enfants de 5 à 17 ans, combien ne vont pas à l'école ?",
            "Enfants non scolarisés", "integer", "Composition", min=0, max=25),
    Feature("femme_enceinte_allaitante", "Le ménage a-t-il une femme enceinte ou allaitante ?", "Femme enceinte / allaitante",
            "categorical", "Composition", OUI_NON),
    Feature("membre_handicap", "Un membre du ménage vit-il avec un handicap ?", "Membre handicapé", "categorical",
            "Composition", OUI_NON),
    Feature("membre_maladie_chronique", "Un membre du ménage a-t-il une maladie chronique ?", "Maladie chronique",
            "categorical", "Composition", OUI_NON),
    Feature("enfant_non_accompagne_heberge", "Le ménage héberge-t-il un enfant non accompagné ou séparé ?",
            "Enfant non accompagné hébergé", "categorical", "Composition", OUI_NON),
    # ------------------------------------------------------------------ logement et biens
    Feature("type_logement", "Type de logement actuel", "Logement", "categorical", "Logement et biens", {
        "maison_correcte": "Maison en bon état", "maison_endommagee": "Maison endommagée",
        "heberge_famille": "Hébergé par une famille", "abri_fortune": "Abri de fortune", "sans_abri": "Sans abri",
    }),
    Feature("statut_occupation", "Statut d'occupation du logement", "Occupation du logement", "categorical",
            "Logement et biens", {
                "proprietaire": "Propriétaire", "locataire": "Locataire", "heberge_gratuit": "Hébergé gratuitement",
                "site_deplaces": "Site ou centre collectif de déplacés",
            }),
    Feature("nombre_pieces", "Nombre de pièces pour dormir", "Pièces pour dormir", "integer", "Logement et biens",
            min=0, max=15, hint="0 si le ménage dort à la belle étoile"),
    Feature("articles_essentiels", "Combien de ces 5 articles le ménage possède-t-il en quantité suffisante ?",
            "Articles essentiels (sur 5)", "integer", "Logement et biens", min=0, max=5,
            hint="Couvertures, nattes, bidons, ustensiles de cuisine, moustiquaires"),
    Feature("possede_betail", "Le ménage possède-t-il du bétail (au moins 2 chèvres ou équivalent) ?", "Possède du bétail",
            "categorical", "Logement et biens", OUI_NON),
    Feature("acces_terre_cultivable", "Le ménage a-t-il accès à une terre qu'il peut cultiver cette saison ?",
            "Accès à une terre cultivable", "categorical", "Logement et biens", OUI_NON),
    Feature("toit_etanche", "Le toit du logement protège-t-il de la pluie ?", "Toit étanche", "categorical",
            "Logement et biens", OUI_NON),
    Feature("source_eclairage", "Principale source d'éclairage du ménage", "Éclairage", "categorical", "Logement et biens", {
        "electricite": "Électricité (réseau)", "solaire": "Panneau ou lampe solaire",
        "lampe_torche": "Lampe torche, bougie, lampe à pétrole", "aucune": "Aucune",
    }),
    Feature("possede_telephone", "Un membre du ménage possède-t-il un téléphone ?", "Possède un téléphone", "categorical",
            "Logement et biens", OUI_NON),
    # ------------------------------------------------------------------ eau et hygiène
    Feature("source_eau", "Principale source d'eau de boisson", "Source d'eau", "categorical", "Eau et hygiène", {
        "robinet": "Robinet / borne fontaine", "puits_protege": "Puits ou source aménagés",
        "source_non_protegee": "Puits ou source non protégés", "eau_surface": "Eau de surface (rivière, lac)",
    }),
    Feature("temps_collecte_eau_min", "Temps nécessaire pour aller chercher l'eau et revenir (minutes)",
            "Temps de collecte de l'eau (min)", "integer", "Eau et hygiène", min=0, max=600),
    Feature("type_latrine", "Type de latrine utilisée", "Latrine", "categorical", "Eau et hygiène", {
        "privee": "Latrine du ménage", "partagee": "Latrine partagée", "aucune": "Aucune (défécation à l'air libre)",
    }),
    Feature("savon_disponible", "Y a-t-il du savon dans le ménage aujourd'hui ?", "Savon disponible", "categorical",
            "Eau et hygiène", OUI_NON),
    # ------------------------------------------------------------------ santé et nutrition
    Feature("enfant_malade_2_semaines", "Un enfant de moins de 5 ans a-t-il eu la diarrhée ou de la fièvre ces 2 dernières semaines ?",
            "Enfant malade (2 semaines)", "categorical", "Santé et nutrition", OUI_NON,
            hint="« Non » si le ménage n'a pas d'enfant de moins de 5 ans"),
    Feature("acces_soins", "La dernière fois qu'un membre a eu besoin de soins, a-t-il pu les obtenir ?", "Accès aux soins",
            "categorical", "Santé et nutrition", {
                "oui": "Oui", "non_cout": "Non, trop cher", "non_distance": "Non, trop loin",
                "non_insecurite": "Non, à cause de l'insécurité", "pas_besoin": "Pas de besoin de soins",
            }),
    Feature("vaccination_enfants_a_jour", "Les enfants de moins de 5 ans ont-ils tous leur carte de vaccination à jour ?",
            "Vaccination à jour", "categorical", "Santé et nutrition", {
                "oui": "Oui", "non": "Non", "pas_enfant": "Pas d'enfant de moins de 5 ans",
            }),
    Feature("enfant_malnutrition_depiste", "Un enfant du ménage a-t-il été dépisté malnutri (périmètre brachial jaune ou rouge) ?",
            "Enfant dépisté malnutri", "categorical", "Santé et nutrition", OUI_NON),
    # ------------------------------------------------------------------ alimentation (score de consommation alimentaire)
    jours("jours_cereales", "céréales, tubercules (maïs, manioc, riz, banane plantain)", "Jours céréales / tubercules"),
    jours("jours_legumineuses", "légumineuses (haricots, petits pois, arachides)", "Jours légumineuses"),
    jours("jours_lait", "lait ou produits laitiers", "Jours lait"),
    jours("jours_viande_poisson", "viande, poisson ou œufs", "Jours viande / poisson / œufs"),
    jours("jours_legumes", "légumes ou feuilles (sombé, amarante)", "Jours légumes"),
    jours("jours_fruits", "fruits", "Jours fruits"),
    jours("jours_huile", "huile ou graisse", "Jours huile"),
    jours("jours_sucre", "sucre ou miel", "Jours sucre"),
    Feature("stock_nourriture_jours", "Pour combien de jours le ménage a-t-il des réserves de nourriture ?",
            "Réserves de nourriture (jours)", "integer", "Alimentation", min=0, max=365, hint="0 si aucune réserve"),
    Feature("repas_par_jour", "Nombre de repas pris par jour par les adultes (7 derniers jours)", "Repas par jour",
            "integer", "Alimentation", min=1, max=3),
    Feature("jours_sans_nourriture_7j", "Au cours des 7 derniers jours, combien de jours le ménage n'a-t-il rien eu à manger ?",
            "Jours sans nourriture (7 j)", "integer", "Alimentation", min=0, max=7),
    # ------------------------------------------------------------------ stratégies d'adaptation
    rcsi("rcsi_aliments_moins_chers", "consommer des aliments moins préférés ou moins chers", "Aliments moins chers (jours)"),
    rcsi("rcsi_emprunt_nourriture", "emprunter de la nourriture ou compter sur l'aide de proches", "Emprunt de nourriture (jours)"),
    rcsi("rcsi_reduction_portions", "réduire la taille des portions", "Portions réduites (jours)"),
    rcsi("rcsi_adultes_pour_enfants", "réduire la consommation des adultes pour nourrir les enfants",
         "Adultes mangent moins (jours)"),
    rcsi("rcsi_moins_de_repas", "réduire le nombre de repas", "Moins de repas (jours)"),
    lcs("lcs_emprunt_argent", "emprunter de l'argent ou acheter à crédit", "Emprunt d'argent / crédit"),
    lcs("lcs_vente_biens_menagers", "vendre des biens du ménage (meubles, radio, ustensiles)", "Vente de biens du ménage"),
    lcs("lcs_consommation_semences", "consommer les semences prévues pour la prochaine saison", "Consommation des semences"),
    lcs("lcs_reduction_depenses_sante", "réduire les dépenses de santé ou d'éducation", "Dépenses de santé réduites"),
    lcs("lcs_retrait_enfants_ecole", "retirer des enfants de l'école", "Enfants retirés de l'école"),
    lcs("lcs_vente_derniere_femelle", "vendre la dernière femelle reproductrice (chèvre, vache)", "Vente de la dernière femelle"),
    lcs("lcs_vente_maison_terre", "vendre la maison ou la terre", "Vente de la maison ou de la terre"),
    lcs("lcs_mendicite", "mendier", "Mendicité"),
    lcs("lcs_travail_enfants", "envoyer des enfants travailler", "Travail des enfants"),
    # ------------------------------------------------------------------ moyens d'existence
    Feature("source_revenu", "Principale source de revenu du ménage", "Source de revenu", "categorical", "Moyens d'existence", {
        "salaire_regulier": "Salaire régulier", "petit_commerce": "Petit commerce", "agriculture": "Agriculture",
        "travail_journalier": "Travail journalier", "aucun_revenu": "Aucun revenu / mendicité",
    }),
    Feature("membres_avec_revenu", "Combien de membres du ménage ont gagné de l'argent au cours des 30 derniers jours ?",
            "Membres avec un revenu", "integer", "Moyens d'existence", min=0, max=20),
    Feature("perte_revenu_6_mois", "Le ménage a-t-il perdu sa principale source de revenu au cours des 6 derniers mois ?",
            "Perte de revenu (6 mois)", "categorical", "Moyens d'existence", OUI_NON),
    Feature("revenu_30j_usd", "Revenu total du ménage au cours des 30 derniers jours (en dollars US)", "Revenu 30 jours (USD)",
            "integer", "Moyens d'existence", min=0, max=5000, hint="Convertir les francs congolais en dollars"),
    Feature("part_depenses_alimentation_pct", "Part des dépenses du ménage consacrée à la nourriture (%)",
            "Part des dépenses pour la nourriture (%)", "integer", "Moyens d'existence", min=0, max=100),
    Feature("dette_usd", "Montant des dettes actuelles du ménage (en dollars US)", "Dette (USD)", "integer",
            "Moyens d'existence", min=0, max=20000, hint="0 si aucune dette"),
    Feature("epargne_disponible", "Le ménage a-t-il de l'épargne (argent mis de côté, tontine, mobile money) ?",
            "Épargne disponible", "categorical", "Moyens d'existence", OUI_NON),
    # ------------------------------------------------------------------ protection
    Feature("incident_securite_3_mois", "Un membre du ménage a-t-il subi un incident de sécurité ces 3 derniers mois ?",
            "Incident de sécurité (3 mois)", "categorical", "Protection", OUI_NON,
            hint="Vol, agression, extorsion, enlèvement, violence"),
    Feature("biens_perdus_crise", "Le ménage a-t-il perdu des biens à cause de la crise (pillage, incendie, destruction) ?",
            "Biens perdus à cause de la crise", "categorical", "Protection", OUI_NON),
    Feature("membre_disparu_detenu", "Un membre du ménage a-t-il disparu, été enlevé ou détenu depuis le début de la crise ?",
            "Membre disparu ou détenu", "categorical", "Protection", OUI_NON),
    Feature("documents_identite", "Le chef de ménage possède-t-il une pièce d'identité ou un document d'électeur ?",
            "Pièce d'identité", "categorical", "Protection", OUI_NON),
    Feature("restriction_mouvements", "Les membres du ménage évitent-ils certains lieux (champs, marché) par peur pour leur sécurité ?",
            "Restriction des mouvements", "categorical", "Protection", OUI_NON),
    # ------------------------------------------------------------------ assistance
    Feature("assistance_3_derniers_mois", "Le ménage a-t-il reçu une assistance humanitaire au cours des 3 derniers mois ?",
            "Assistance reçue (3 mois)", "categorical", "Assistance", OUI_NON),
    Feature("type_assistance_recue", "Quel type d'assistance ?", "Type d'assistance reçue", "categorical", "Assistance", {
        "vivres": "Vivres", "cash": "Cash", "articles_menagers": "Articles ménagers", "abri": "Abri", "autre": "Autre",
    }, relevant=("assistance_3_derniers_mois", frozenset({"oui"}))),
]

FEATURES_BY_NAME = {f.name: f for f in FEATURES}
FEATURES_BY_KOBO = FEATURES_BY_NAME
FEATURE_NAMES = [f.name for f in FEATURES if f.modele]      # variables d'entrée du modèle
CATEGORICAL = [n for n in FEATURE_NAMES if FEATURES_BY_NAME[n].kind == "categorical"]
NUMERIC = [n for n in FEATURE_NAMES if FEATURES_BY_NAME[n].kind == "integer"]
SHORT_LABELS = {f.name: f.short for f in FEATURES}
THEMES = {f.name: f.theme for f in FEATURES}
# Types d'assistance que le second modèle peut recommander (besoin principal du ménage)
ASSISTANCES = {
    "vivres": "Vivres / cash alimentaire",
    "moyens_existence": "Moyens d'existence (semences, activité génératrice de revenu)",
    "abri_ame": "Abri et articles ménagers essentiels",
    "education": "Éducation (kits et frais scolaires)",
    "sante_nutrition": "Santé et nutrition",
    "eau_hygiene": "Eau, hygiène et assainissement",
    "protection": "Protection",
}
ZONE_FIELD = "localite"  # question utilisée pour la répartition du budget par zone

# Questions d'identification : stockées et affichées, jamais utilisées par le modèle.
IDENTITY_FIELDS = {
    "nom_chef_menage": "Nom du chef de ménage",
    "telephone": "Téléphone",
    "village": "Village / quartier",
    "enumerateur": "Enquêteur",
    "gps": "Coordonnées GPS",
}

# Contraintes entre questions : (question qui porte la contrainte, contrainte Kobo, règle vérifiée par l'API, message)
COHERENCE = [
    ("personnes_60_plus", ". + ${enfants_moins_5} + ${enfants_5_17} <= ${taille_menage}",
     lambda v: v["enfants_moins_5"] + v["enfants_5_17"] + v["personnes_60_plus"] <= v["taille_menage"],
     "enfants et personnes âgées dépassent la taille du ménage"),
    ("enfants_non_scolarises", ". <= ${enfants_5_17}",
     lambda v: v["enfants_non_scolarises"] <= v["enfants_5_17"],
     "ne peut pas dépasser le nombre d'enfants de 5 à 17 ans"),
]


class ValidationError(ValueError):
    pass


def libelle(name: str, value) -> str:
    """Libellé affiché d'une réponse (code -> texte), pour le dashboard et les explications."""
    f = FEATURES_BY_NAME.get(name)
    if value is None or (isinstance(value, float) and value != value):
        return "non applicable"
    if f and f.choices:
        return f.choices.get(str(value), str(value))
    return str(int(value)) if isinstance(value, (int, float)) else str(value)


def _est_posee(f: Feature, valeurs: dict) -> bool:
    if f.relevant is None:
        return True
    question, attendues = f.relevant
    return valeurs.get(question) in attendues


def normalize_features(values: dict) -> dict:
    """Convertit les réponses Kobo (codes ou libellés) en valeurs du modèle.

    Une question conditionnelle non posée vaut None.
    Lève ValidationError avec la liste de tous les problèmes trouvés.
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
    if not errors:
        errors += [f"{nom}: {message}" for nom, _, regle, message in COHERENCE if not regle(out)]
    if errors:
        raise ValidationError("; ".join(errors))
    return out
