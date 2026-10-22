"""Ingénierie des variables, partagée par l'entraînement et l'API.

Les indicateurs reprennent les critères de vulnérabilité IFRC utilisés pour
le ciblage. Ils sont dérivés des réponses brutes, donc aucune question
supplémentaire n'est nécessaire dans le formulaire.
"""
import pandas as pd

# indicateur dérivé -> question brute dont il provient (sert à regrouper l'explication)
ENGINEERED_SOURCES = {
    "Chef_Mineur": "Age_Chef_Menage",
    "Chef_Age_60plus": "Age_Chef_Menage",
    "Grand_Menage_8plus": "Taille_Menage",
    "Un_Repas_Jour": "Nombre_Repas_Jour",
}


def add_vulnerability_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Chef_Mineur"] = (df["Age_Chef_Menage"] < 18).astype(int)
    df["Chef_Age_60plus"] = (df["Age_Chef_Menage"] > 60).astype(int)
    df["Grand_Menage_8plus"] = (df["Taille_Menage"] > 7).astype(int)
    df["Un_Repas_Jour"] = (df["Nombre_Repas_Jour"] <= 1).astype(int)
    return df
