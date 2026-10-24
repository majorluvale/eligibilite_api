# Notebooks d'analyse

| Notebook | Données | Ce qu'il montre |
|---|---|---|
| `ciblage_avec_ml_v3_desequilibre.ipynb` | `dataset_humanitaire_complexe_100k.csv` | Classes déséquilibrées (13,7 % d'éligibles) : pondération, choix du seuil (F2), corrections de la v2. |
| `ciblage_avec_ml_v4_msna.ipynb` | `msna_household_targeting_dataset.csv` | Données de type MSNA : faux manquants, sélection de variables (11 questions sur 30), seuil fondé sur le coût des erreurs, plafond dû au bruit des étiquettes. |
| `generation_donnees_ciblage_v5.ipynb`, `ciblage_avec_ml_v5.ipynb` | `data/ciblage_v5.db` | **Abandonnée.** Le modèle apprenait à reproduire la grille Excel de l'agent et son seuil : des scores très hauts, mais le modèle recopiait la grille et n'apportait rien de plus qu'Excel. |
| `generation_donnees_ciblage_v6.ipynb` | génère `data/ciblage_v6.csv` et `data/ciblage_v6.db` | Enquête ménage complexe (72 variables) et décision de la **vérification approfondie** (visite à domicile et comité de ciblage). La grille Excel est calculée, mais seulement comme point de comparaison. |
| `ciblage_avec_ml_v6.ipynb` | `data/ciblage_v6.csv` | Exploration, préparation, 7 modèles comparés (KNN, régression logistique, SVM, arbre de décision, Random Forest, Gradient Boosting, XGBoost), validation croisée, SHAP, comparaison avec la grille Excel. **C'est l'approche déployée dans l'API** : `training/train.py` entraîne le même XGBoost. |

## Exécution

- **v6** : lancer d'abord `generation_donnees_ciblage_v6.ipynb` (il crée le dossier `data/`, qui n'est pas versionné), puis `ciblage_avec_ml_v6.ipynb`. Les deux s'exécutent depuis ce dossier, avec l'environnement de l'API (`pip install -r ../requirements-dev.txt`). Le notebook principal prend une quinzaine de minutes (recherches GridSearch).
- **v3 et v4** ont été exécutés depuis le dossier parent du projet (`PROJET_FIN_FORMATION/`), là où se trouvent leurs fichiers CSV. Ces fichiers ne sont pas versionnés, car trop volumineux. Les résultats sont conservés dans les notebooks.
- `generateur_ciblage_v5.py` est conservé : la v6 réutilise sa grille Excel pour la comparaison.
