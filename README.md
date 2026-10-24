# API d'éligibilité des ménages — KoboToolbox + ML explicable

Backend FastAPI qui :

1. **reçoit une copie de chaque soumission KoboToolbox** (REST Service / webhook) ;
2. **prédit l'éligibilité du ménage** avec un modèle XGBoost entraîné sur les décisions de la **vérification approfondie** (visite à domicile et comité de ciblage), à partir des 72 réponses de l'enquête ménage. La grille Excel de l'ONG, son score et son seuil ne sont **pas** donnés au modèle ;
3. **explique la décision** : effet de chaque réponse et de chaque thème du questionnaire (contributions SHAP) ;
4. permet à un·e superviseur·e de **voir les données du ménage et la décision, puis de l'approuver ou de la désapprouver**. Un commentaire est obligatoire en cas de désaccord, et chaque revue est historisée ;
5. **sélectionne les ménages à servir quand le budget est limité**, par indice de priorité du modèle, au global ou par zone.

```
KoboCollect ──soumission──▶ KoboToolbox ──REST Service (JSON)──▶ POST /api/kobo/webhook
                                                                     │
                                       schema.py : questions du formulaire = variables du modèle
                                       modèle    : XGBoost (72 variables) + explication SHAP
                                                                     ▼
                                     base de données (SQLite / Postgres) ──▶ /dashboard
                                                                               ├─ liste, filtres, zone grise
                                                                               ├─ fiche ménage + explication
                                                                               ├─ approuver / désapprouver
                                                                               ├─ sélection sous budget
                                                                               └─ export CSV
```

## Pourquoi un modèle et pas seulement la grille Excel ?

- Dans un scénario simple (une dizaine de questions oui/non), la grille Excel fait le travail.
- Avec plus de 70 variables, une grille de points additionnés ne peut pas tout combiner, et l'humain peut se tromper. Le modèle cherche lui-même les relations entre les réponses : par exemple, un même revenu ne pèse pas pareil dans un ménage de 3 personnes et dans un ménage de 10 personnes sans réserves de nourriture.
- Le modèle ne reproduit pas la grille : entraîné à la recopier (version 5), il obtenait des scores très hauts, mais il n'apportait rien de plus qu'Excel. Cette approche a été abandonnée.
- Le modèle aide à décider à grande échelle, il ne remplace pas la décision : chaque décision reste revue par un superviseur dans le dashboard.

## Contenu

| Chemin | Rôle |
|---|---|
| `app/schema.py` | **Source unique** des questions : nom, libellé, choix, thème, questions conditionnelles, contrôles de cohérence |
| `app/ml.py` | Modèle XGBoost, prédiction et explication SHAP (par réponse et par thème) |
| `app/main.py` | Routes API + dashboard |
| `training/train.py` | Entraînement reproductible (même préparation et mêmes paramètres que le notebook v6) |
| `model/` | Modèle déployé (`model.joblib`) et métriques (`metadata.json`) |
| `kobo/build_xlsform.py` | Génère le formulaire Kobo depuis `schema.py` (un groupe par thème) |
| `kobo/formulaire_ciblage_menages.xlsx` | **Formulaire à importer dans KoboToolbox** |
| `scripts/simulate_kobo.py` | Envoie des ménages simulés au webhook (démo) |
| `tests/` | Tests de l'API, du webhook, de la revue, de la sélection et de la cohérence formulaire ↔ modèle |
| `notebooks/` | Analyses v3 à v6 ; **v6 = modèle déployé** (voir `notebooks/README.md`) |

## Sélection sous contrainte budgétaire

Quand il y a plus de ménages éligibles que le budget n'en permet (par exemple 10 000 éligibles et 3 000 places), la page **Sélection** du dashboard classe les éligibles et produit la liste.

- **Indice de priorité** : le log-odds du modèle, c'est-à-dire la somme des effets de toutes les réponses du ménage (les mêmes que dans « Pourquoi le modèle a décidé ainsi »). Plus il est élevé, plus le profil du ménage ressemble à celui des ménages retenus par la vérification.
  - On ne classe **pas** par probabilité d'éligibilité : elle plafonne à 100 % pour beaucoup de ménages et ne les départage plus. L'indice continue de les séparer.
  - La grille Excel n'intervient pas dans ce classement.
- **Deux modes de répartition** :
  - **globale** : les N ménages les plus prioritaires, toutes zones confondues ;
  - **proportionnelle par zone** : chaque zone reçoit un quota proportionnel à son nombre d'éligibles (méthode du plus fort reste), puis on prend les plus prioritaires dans chaque zone.
- **Option** : n'inclure que les ménages dont la décision a déjà été revue par un humain.
- Chaque liste est un **instantané** enregistré : auteur, date, paramètres, version du modèle. Elle affiche :
  - la couverture par zone ;
  - les ménages **juste sous la ligne de coupure** (à examiner en cas de contestation ou de place libérée) ;
  - un signal si un ménage retenu a été déclaré non éligible après la création de la liste.
- Export CSV pour la distribution : rang, ménage, téléphone, village, zone, indice, principaux facteurs.
- En cas d'égalité, le ménage enquêté le plus tôt passe d'abord.

## Le modèle

Analyse complète : `notebooks/ciblage_avec_ml_v6.ipynb`. Sept modèles y sont comparés (KNN, régression logistique, SVM, arbre de décision, Random Forest, Gradient Boosting, XGBoost), et **XGBoost est le plus performant**.

- **Données** : 100 000 ménages simulés, 72 variables de l'enquête ménage (profil, composition, logement, eau et hygiène, santé, consommation alimentaire, stratégies d'adaptation, moyens d'existence, protection, assistance). Pas de GPS ni de données externes.
- **Cible** : décision de la vérification approfondie (30 % d'éligibles).
- **Colonnes retirées avant l'entraînement** : identifiants, localité (elle sert seulement à répartir le budget), et la grille de l'agent (`score_grille_excel`, `decision_grille_excel`), comme la colonne `Score_Vuln` de la première version, qui provoquait l'overfitting.
- **Préparation** : questions conditionnelles non posées → -1 ou `non_applicable`, choix encodés avec LabelEncoder ; découpage 80 / 10 / 10 stratifié.
- **XGBoost** : 3 000 arbres de profondeur 3, taux d'apprentissage 0,05 (paramètres choisis par GridSearchCV), seuil 0,5.
- **Explication** : contributions TreeSHAP, qui s'additionnent exactement : `log-odds = ménage moyen + somme des effets des réponses`, et probabilité = sigmoïde(log-odds).

Résultats sur le jeu de test (10 000 ménages, mêmes ménages pour les trois méthodes) :

| | Accuracy | F1 | Rappel | Précision |
|---|---|---|---|---|
| Grille Excel de l'agent (seuil 40 points) | 78,8 % | 62,3 % | 58,5 % | 66,8 % |
| Régression logistique (validation) | 94,9 % | 91,4 % | 91,2 % | 91,6 % |
| **XGBoost (déployé)** | **98,1 %** | **96,9 %** | **96,2 %** | **97,6 %** |

- Validation croisée en 5 parties : toutes les métriques restent au-dessus de 95 % sur chaque partie (voir `model/metadata.json` et le notebook).
- Sur les 1 246 ménages éligibles que la grille Excel oublie, le modèle en retrouve **93,7 %**. Sur les 873 ménages que la grille retient à tort, il en écarte **96,6 %**.
- **Zone grise** (probabilité entre 20 % et 80 %) : 6,9 % des ménages, mais 87,7 % des erreurs. Ces ménages sont signalés pour être revus en priorité.
- Les thèmes qui pèsent le plus sont l'alimentation, les moyens d'existence et les stratégies d'adaptation. Selon les valeurs SHAP, les questions qui pèsent le plus sont le revenu, la taille du ménage et les réserves de nourriture, ce qui correspond au revenu par personne, une variable qui ne lui a pas été donnée.

Les données sont simulées : sur de vraies décisions de vérification, qui contiennent plus de jugement, les scores seront plus bas, et la revue humaine prend tout son sens.

Ré-entraîner (les données v6 sont générées automatiquement si elles sont absentes) :

```bash
python -m training.train                          # données v6, avec validation croisée (quelques minutes)
python -m training.train --data mes_menages.csv   # vos propres données : questions de schema.py + eligible (0/1)
```

## Lancer en local

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS : source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env            # puis modifier les mots de passe et jetons
uvicorn app.main:app --reload
```

- Dashboard : http://localhost:8000/dashboard (comptes définis dans `DASHBOARD_USERS`)
- Documentation interactive de l'API : http://localhost:8000/docs
- Données de démonstration : `python scripts/simulate_kobo.py --token <KOBO_WEBHOOK_TOKEN> -n 40`
- Tests : `pytest`

## Mettre en place KoboToolbox

1. **Importer le formulaire** : sur https://kf.kobotoolbox.org (ou `eu.kobotoolbox.org`), cliquer sur *Nouveau → Importer un XLSForm* → `kobo/formulaire_ciblage_menages.xlsx` → *Déployer*.
   - **Mettre à jour un projet existant** : ouvrir le projet, puis *Formulaire → ⋯ → Remplacer le formulaire* → `kobo/formulaire_ciblage_menages.xlsx` → *Redéployer*. Le REST Service est conservé.
   - Le formulaire compte 73 questions (72 pour le modèle, plus la localité) réparties en 10 groupes, un par thème. Les questions conditionnelles (durée et nombre de déplacements, type d'assistance) et les contrôles de cohérence (enfants et personnes âgées ≤ taille du ménage, enfants non scolarisés ≤ enfants de 5 à 17 ans) sont vérifiés par Kobo dès la saisie, et à nouveau par l'API.
   - **Attention** : le nouveau formulaire n'est pas compatible avec l'ancien (questions différentes). Les ménages déjà évalués avec l'ancien formulaire restent consultables, mais ne sont pas repris dans les sélections.
   - Pour modifier le formulaire, changer `app/schema.py`, puis lancer `python kobo/build_xlsform.py` et réimporter. Ne pas renommer les questions ni les codes de choix directement dans Kobo : l'API ne les reconnaîtrait plus.
   - Les questions peuvent être déplacées entre groupes : l'API ignore les préfixes de groupe.
2. **Brancher le webhook** : dans le projet, *Paramètres → REST Services → Enregistrer un nouveau service* :
   - URL du point de terminaison : `https://<votre-app>/api/kobo/webhook?token=<KOBO_WEBHOOK_TOKEN>`
     (vous pouvez aussi mettre le jeton dans un en-tête HTTP personnalisé `X-Webhook-Token`)
   - Type : **JSON**
3. **Collecter** avec KoboCollect ou Enketo. Chaque soumission apparaît dans le dashboard quelques secondes après son envoi au serveur.
4. **Rattrapage (optionnel)** : avec `KOBO_API_TOKEN` (*Paramètres du compte → Sécurité → Clé API*) et `KOBO_ASSET_UID` (l'identifiant `aXXXXXXXX` dans l'URL du formulaire), le bouton **Synchroniser Kobo** du dashboard importe toutes les soumissions déjà présentes. Les doublons sont ignorés.

Comportement du webhook :
- une soumission renvoyée par Kobo n'est pas dupliquée (clé = `_id` Kobo) ;
- si une soumission est **modifiée** dans Kobo, le modèle la réévalue et toute revue existante est annulée, avec une trace dans l'historique ;
- si le consentement est refusé, la soumission est ignorée ;
- si une réponse est invalide (choix inconnu, âge hors bornes), la réponse est un `422` avec le détail, visible dans le journal REST Services de Kobo.

## Héberger gratuitement

**Option recommandée : Render (web) + Neon (Postgres)**

1. Créer une base gratuite sur https://neon.tech et copier l'URL de connexion (`postgresql://...?sslmode=require`).
   Elle est indispensable : le disque gratuit de Render est effacé à chaque redémarrage, donc SQLite y perdrait les revues.
2. Pousser ce dossier sur GitHub, puis sur https://render.com : *New → Blueprint* et choisir le dépôt. `render.yaml` est détecté automatiquement.
3. Renseigner `DATABASE_URL` (Neon) et `DASHBOARD_USERS`. `SECRET_KEY` et `KOBO_WEBHOOK_TOKEN` sont générés automatiquement : copier ce dernier dans l'URL du REST Service Kobo.

Limite du plan gratuit : le service s'endort après 15 minutes sans trafic, et le réveil prend environ 1 minute. La première soumission après une pause peut donc échouer côté Kobo. Dans ce cas, relancez-la depuis le journal *REST Services* de Kobo, ou cliquez sur **Synchroniser Kobo**.

**Alternative : Hugging Face Spaces** (Docker, gratuit, ne s'endort pas aussi vite) : créer un Space *Docker*, y copier ce dossier, ajouter `app_port: 7860` dans l'en-tête du README du Space et définir les mêmes variables dans *Settings → Variables and secrets*. Utilisez aussi Neon pour la base.

## API

| Méthode | Route | Auth | Rôle |
|---|---|---|---|
| POST | `/api/kobo/webhook` | jeton | Réception d'une soumission Kobo |
| POST | `/api/predict` | — | Prédiction + explication (par réponse et par thème), sans enregistrement |
| GET | `/api/modele` | — | Métriques et paramètres du modèle |
| GET | `/api/beneficiaires` | Basic | Liste (`statut`, `decision`, `limit`, `offset`) |
| GET | `/api/beneficiaires/{id}` | Basic | Détail : réponses, explication, historique des revues |
| POST | `/api/beneficiaires/{id}/revision` | Basic | `{"approuve": false, "commentaire": "..."}` |
| POST | `/api/kobo/sync` | Basic | Import des soumissions via l'API Kobo |
| POST | `/api/selections` | Basic | `{"nombre": 3000, "repartition": "globale" \| "proportionnelle_zone", "inclure_non_revus": true, "nom": "..."}` |
| GET | `/api/selections/{id}` | Basic | Liste retenue, classée par score estimé |
| GET | `/health` | — | Supervision |

## Pistes suivantes

- Ré-entraîner le modèle avec les décisions corrigées par les superviseur·e·s (export CSV du dashboard), pour passer de données simulées à des données réelles.
- Renvoyer la décision finale dans Kobo (champ calculé, ou mise à jour de la soumission via l'API Kobo `PATCH /data/bulk/`).
- **Détecter les anomalies** : réponses incohérentes entre elles, enquêteur dont les décisions s'écartent nettement du modèle, ménages très différents de ceux vus à l'entraînement.
