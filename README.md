# API d'éligibilité des ménages — KoboToolbox + ML explicable

Backend FastAPI qui :

1. **reçoit une copie de chaque soumission KoboToolbox** (REST Service / webhook) ;
2. **prédit si le ménage est éligible** avec le modèle entraîné (classes déséquilibrées : 13,7 % d'éligibles) ;
3. **explique la décision** : contribution de chaque critère, dans un dashboard ;
4. permet à un·e superviseur·e de **voir les données du ménage et la décision, puis de l'approuver ou de la désapprouver**. Un commentaire est obligatoire en cas de désaccord, et chaque revue est historisée.

```
KoboCollect ──soumission──▶ KoboToolbox ──REST Service (JSON)──▶ POST /api/kobo/webhook
                                                                     │
                                       schema.py : codes Kobo → variables du modèle
                                       modèle    : score + seuil + contributions
                                                                     ▼
                                     base de données (SQLite / Postgres) ──▶ /dashboard
                                                                               ├─ liste, filtres, zone grise
                                                                               ├─ fiche ménage + explication
                                                                               ├─ approuver / désapprouver
                                                                               └─ export CSV
```

## Contenu

| Chemin | Rôle |
|---|---|
| `app/schema.py` | **Source unique** des questions : nom Kobo, libellé, codes des choix, valeur d'entraînement |
| `app/features.py` | Indicateurs de vulnérabilité IFRC dérivés des réponses (chef mineur, chef de plus de 60 ans, 8 personnes ou plus, 1 repas par jour) |
| `app/ml.py` | Chargement du modèle, prédiction, explication additive (exacte) |
| `app/main.py` | Routes API + dashboard |
| `training/train.py` | Entraînement reproductible, comparaison des modèles, choix du seuil |
| `model/` | Modèle déployé (`model.joblib`) et métriques (`metadata.json`) |
| `kobo/build_xlsform.py` | Génère le formulaire Kobo depuis `schema.py` |
| `kobo/formulaire_ciblage_menages.xlsx` | **Formulaire à importer dans KoboToolbox** |
| `scripts/simulate_kobo.py` | Envoie des ménages simulés au webhook (démo) |
| `tests/` | Tests de l'API, du webhook, de la revue, de la sélection et de la cohérence formulaire ↔ modèle |
| `notebooks/` | Analyses v3, v4 et v5 : déséquilibre, données MSNA, grille de points d'une ONG et sélection sous budget (voir `notebooks/README.md`) |

## Sélection sous contrainte budgétaire

Quand il y a plus de ménages éligibles que le budget n'en permet (par exemple 10 000 éligibles et 3 000 places), la page **Sélection** du dashboard classe les éligibles et produit la liste.

- **Score de priorité** : un **second modèle**, le modèle de priorité, apprend à prédire le **score que l'agent calculait avec sa grille Excel** (ici `Score_Vuln`). Les éligibles sont classés par ce score prédit.
  - Sur les données v5 (grille avec exclusions et inclusions automatiques), il retrouve **96 %** de la liste qu'aurait produite l'agent pour un budget couvrant 30 % des éligibles, contre 83 à 85 % en classant avec le modèle de décision. Voir `notebooks/ciblage_avec_ml_v5.ipynb`.
  - Si l'historique ne contient pas de score de l'agent, le classement se fait par repli sur l'indice du modèle de décision (`base + somme des contributions`).
  - On ne classe **pas** par probabilité d'éligibilité : elle mesure la certitude du modèle, pas l'intensité des besoins, et elle sature à 1.
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

Voir `../ciblage_avec_ml_v3_desequilibre.ipynb` pour l'analyse complète. Résumé sur le jeu de test (15 000 ménages) :

| | v2 (XGBoost, seuil 0,5) | v3 (régression logistique pondérée, seuil 0,45) |
|---|---|---|
| Rappel (éligibles retrouvés) | 86,9 % | **98,5 %** |
| Éligibles exclus | 269 | **31** |
| Précision | 88,6 % | 75,5 % |
| F2 | 0,872 | **0,928** |
| PR-AUC | 0,960 | **0,965** |

- **Déséquilibre** : pondération des classes, plus un seuil choisi sur la validation pour maximiser le F2. Le F2 privilégie le rappel, car exclure un ménage vulnérable est plus grave qu'en inclure un de trop. SMOTE et le sous-échantillonnage ont été testés et n'apportent rien de plus.
- **Plafond** : les données simulées contiennent un bruit aléatoire dans l'étiquette. Le modèle atteint le niveau d'un « oracle » qui connaîtrait le score de vulnérabilité exact. Les erreurs restantes se concentrent près du seuil, et c'est pour cela qu'il y a une revue humaine.
- **Zone grise** : les ménages dont le score est à ±0,30 du seuil (≈ 6 % des ménages, ≈ 43 % des erreurs) sont signalés « à vérifier en priorité ».
- **Explication** : `logit(score) = valeur de base + Σ contributions`. Chaque contribution est regroupée par question du formulaire. Pour la régression logistique elle est exacte (coef × écart à la moyenne) ; pour LightGBM ce sont des valeurs TreeSHAP natives.

Ré-entraîner (modèle de décision, plus modèle de priorité si la colonne `Score_Vuln` du score de l'agent est présente) :

```bash
python -m training.train --data ../dataset_humanitaire_complexe_100k.csv
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
| POST | `/api/predict` | — | Prédiction + explication sans enregistrement (codes Kobo ou valeurs d'entraînement) |
| GET | `/api/modele` | — | Métriques et paramètres du modèle |
| GET | `/api/beneficiaires` | Basic | Liste (`statut`, `decision`, `limit`, `offset`) |
| GET | `/api/beneficiaires/{id}` | Basic | Détail : réponses, explication, historique des revues |
| POST | `/api/beneficiaires/{id}/revision` | Basic | `{"approuve": false, "commentaire": "..."}` |
| POST | `/api/kobo/sync` | Basic | Import des soumissions via l'API Kobo |
| POST | `/api/selections` | Basic | `{"nombre": 3000, "repartition": "globale" \| "proportionnelle_zone", "inclure_non_revus": true, "nom": "..."}` |
| GET | `/api/selections/{id}` | Basic | Liste retenue, classée par indice de priorité |
| GET | `/health` | — | Supervision |

## Pistes suivantes

- Ré-entraîner le modèle avec les décisions corrigées par les superviseur·e·s (export CSV du dashboard), pour passer de données simulées à des données réelles.
- Renvoyer la décision finale dans Kobo (champ calculé, ou mise à jour de la soumission via l'API Kobo `PATCH /data/bulk/`).
