"""Génère le formulaire KoboToolbox (XLSForm) à partir de app/schema.py.

Usage :  python kobo/build_xlsform.py
Sortie : kobo/formulaire_ciblage_menages.xlsx  (à importer dans KoboToolbox)

Le formulaire est généré, pas écrit à la main : les noms des questions et
les codes des choix sont donc toujours ceux que l'API attend.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schema import FEATURES_BY_KOBO  # noqa: E402

OUT = ROOT / "kobo" / "formulaire_ciblage_menages.xlsx"
SURVEY_COLS = ["type", "name", "label", "hint", "required", "constraint", "constraint_message",
               "relevant", "appearance"]

# Ordre et regroupement des questions du modèle dans le formulaire
GROUPS = [
    ("menage", "Composition du ménage",
     ["location", "statut_menage", "sexe_chef_menage", "age_chef_menage", "taille_menage",
      "membre_handicap", "femme_enceinte_allaitante"]),
    ("conditions_vie", "Logement et accès à l'eau", ["type_logement", "source_eau"]),
    ("economie", "Économie et sécurité alimentaire",
     ["source_revenu", "nombre_repas_jour", "dette_contractee_crise"]),
]
# Questions numériques présentées comme un choix (libellés plus clairs pour l'enquêteur)
AS_CHOICES = {"nombre_repas_jour": {"1": "1 repas ou moins", "2": "2 repas", "3": "3 repas ou plus"}}
# Libellés affichés à l'enquêteur quand ils diffèrent des valeurs d'entraînement (les codes ne changent pas)
FORM_LABELS = {"sexe_chef_menage": {"m": "Homme", "f": "Femme"}}
CONSENT = "${consentement} = 'oui'"


def row(**kw) -> dict:
    return kw


def build() -> Path:
    survey, choices = [], []
    shared_lists = {}

    def add_list(list_name: str, items: dict[str, str]):
        if list_name in shared_lists:
            return
        shared_lists[list_name] = items
        choices.extend({"list_name": list_name, "name": k, "label": v} for k, v in items.items())

    add_list("oui_non", {"oui": "Oui", "non": "Non"})

    survey += [
        row(type="start", name="start"),
        row(type="end", name="end"),
        row(type="today", name="today"),
        row(type="deviceid", name="deviceid"),
        row(type="note", name="intro",
            label="**Enquête de ciblage des ménages**\nCe formulaire sert à évaluer l'éligibilité du ménage "
                  "à l'assistance humanitaire. Les réponses sont analysées automatiquement puis revues par l'équipe."),
        row(type="select_one oui_non", name="consentement", required="yes",
            label="Le chef de ménage (ou son représentant) accepte-t-il de répondre à cette enquête ?",
            hint="Lire la déclaration de consentement avant de poser la question."),
        row(type="note", name="fin_sans_consentement", relevant="${consentement} = 'non'",
            label="Le ménage n'a pas donné son consentement. Remerciez la personne et terminez l'entretien."),

        row(type="begin_group", name="identification", label="Identification", relevant=CONSENT),
        row(type="text", name="enumerateur", label="Nom de l'enquêteur", required="yes"),
        row(type="text", name="village", label="Village / quartier", required="yes"),
        row(type="text", name="nom_chef_menage", label="Nom complet du chef de ménage", required="yes"),
        row(type="text", name="telephone", label="Numéro de téléphone du ménage (si disponible)",
            constraint="regex(., '^[+0-9 ]{8,16}$')",
            constraint_message="Numéro invalide : chiffres, espaces ou + uniquement (8 à 16 caractères)",
            appearance="numbers"),
        row(type="geopoint", name="gps", label="Position GPS du ménage"),
        row(type="end_group", name=""),
    ]

    for group, label, names in GROUPS:
        survey.append(row(type="begin_group", name=group, label=label, relevant=CONSENT))
        for name in names:
            f = FEATURES_BY_KOBO[name]
            if name in AS_CHOICES:
                add_list(name, AS_CHOICES[name])
                survey.append(row(type=f"select_one {name}", name=name, label=f.label, hint=f.hint, required="yes"))
            elif f.kind == "integer":
                survey.append(row(type="integer", name=name, label=f.label, hint=f.hint, required="yes",
                                  constraint=f". >= {f.min} and . <= {f.max}",
                                  constraint_message=f"La valeur doit être comprise entre {f.min} et {f.max}"))
            else:
                list_name = "oui_non" if f.choices == {"oui": "Oui", "non": "Non"} else name
                add_list(list_name, FORM_LABELS.get(name, f.choices))
                survey.append(row(type=f"select_one {list_name}", name=name, label=f.label, hint=f.hint,
                                  required="yes"))
        survey.append(row(type="end_group", name=""))

    survey.append(row(type="note", name="fin", relevant=CONSENT,
                      label="Merci. L'éligibilité sera calculée à la réception de la soumission et "
                            "revue par l'équipe de ciblage."))

    wb = Workbook()
    write_sheet(wb.active, "survey", SURVEY_COLS, survey)
    write_sheet(wb.create_sheet(), "choices", ["list_name", "name", "label"], choices)
    write_sheet(wb.create_sheet(), "settings", ["form_title", "form_id", "version", "default_language"], [{
        "form_title": "Ciblage des ménages - éligibilité à l'assistance",
        "form_id": "ciblage_menages_eligibilite",
        "version": datetime.now().strftime("%Y%m%d%H"),
        "default_language": "French (fr)",
    }])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    return OUT


def write_sheet(ws, title: str, cols: list[str], rows: list[dict]):
    ws.title = title
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for r in rows:
        ws.append([r.get(c, "") for c in cols])
    for i, c in enumerate(cols, 1):
        width = max(len(str(r.get(c, ""))) for r in rows + [{c: c}])
        ws.column_dimensions[get_column_letter(i)].width = min(max(12, width + 2), 70)
    ws.freeze_panes = "A2"


if __name__ == "__main__":
    print(f"Formulaire écrit : {build()}")
