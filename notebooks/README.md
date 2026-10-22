# Notebooks d'analyse

| Notebook | Données | Ce qu'il montre |
|---|---|---|
| `ciblage_avec_ml_v3_desequilibre.ipynb` | `dataset_humanitaire_complexe_100k.csv` | Classes déséquilibrées (13,7 % d'éligibles) : pondération, choix du seuil (F2), corrections de la v2. Première version déployée (remplacée par la v5). |
| `ciblage_avec_ml_v4_msna.ipynb` | `msna_household_targeting_dataset.csv` | Données de type MSNA : faux manquants, sélection de variables (11 questions sur 30), seuil fondé sur le coût des erreurs, plafond dû au bruit des étiquettes. |
| `generation_donnees_ciblage_v5.ipynb` | génère `data/ciblage_v5.csv` et `data/ciblage_v5.db` | Simulation du processus de ciblage d'une ONG : grille de points validée localement, inclusions automatiques, exclusions, décision de l'agent dans Excel. |
| `ciblage_avec_ml_v5.ipynb` | `data/ciblage_v5.db` | Reproduire la décision de l'agent (précision 99,1 %) et **sélectionner sous contrainte budgétaire** avec un modèle de priorité (96 % de la liste de l'agent retrouvée). **C'est l'approche déployée dans l'API** : `training/train.py` entraîne les mêmes modèles. |

## Exécution

- **v5** : lancer d'abord `generation_donnees_ciblage_v5.ipynb` (il crée le dossier `data/`, qui n'est pas versionné), puis `ciblage_avec_ml_v5.ipynb`. Les deux s'exécutent depuis ce dossier, avec l'environnement de l'API (`pip install -r ../requirements-dev.txt`).
- **v3 et v4** ont été exécutés depuis le dossier parent du projet (`PROJET_FIN_FORMATION/`), là où se trouvent leurs fichiers CSV. Ces fichiers ne sont pas versionnés, car trop volumineux. Les résultats sont conservés dans les notebooks.
