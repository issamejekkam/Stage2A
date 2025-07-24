# Évaluation automatique des fonctions / postes

Ce projet permet d’évaluer automatiquement des **fonctions ou postes** à partir de leurs cahiers des charges, selon les **responsabilités** ou **compétences**.  
L’algorithme compare chaque cahier des charges à un référentiel de questions d’évaluation, puis attribue un score global selon deux approches :

1. **Analyse lexicale (TF‑IDF)** – comparaison mot à mot.
2. **Analyse sémantique (Sentence Transformers)** – détection des similarités de sens.

> Les résultats sont insérés dans SQL Server et exportés localement pour vérification (`results/` et `log.txt`).

---

## Fonctionnalités principales

- **Double vectorisation** : TF-IDF (lexicale) et Sentence Transformers (sémantique)
- **Connexion à SQL Server** pour interagir avec les questionnaires et stocker les résultats
- **API FastAPI** pour lancer les évaluations
- **Interface Web (.NET 9 WebForm)** pour soumettre les paramètres via formulaire
- **Résultats complets** dans les logs et en JSON

---

## Arborescence du projet

```
├── evaluateFunction.py           # Moteur principal d’évaluation
├── main.py                      # API FastAPI
├── database.py                  # Connexion DB + chargement cahiers des charges
├── pretraitements.py            # Nettoyage, vectorisation
├── similarity.py                # Calculs de similarité
├── requirementsForApi.txt       # Dépendances pour l’API
├── requirementsForEvaluation.txt# Dépendances pour le moteur
├── clients/                     # Fichiers Word ou PDF
└── results/                     # Export des résultats JSON
```

---

## Prérequis

| Logiciel     | Version minimale |
|--------------|------------------|
| Python       | 3.11             |
| .NET SDK     | 9.0              |
| SQL Server   | -                |

---

## Installation

```bash
# 1. Environnement virtuel (optionnel mais recommandé)
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirementsForEvaluation.txt
pip install -r requirementsForApi.txt
```

---

## Paramétrage

- **Connexion SQL Server** : à définir dans `evaluateFunction.py`
- **Répertoire des fichiers** : placez vos fichiers Word/PDF dans `clients/`  
    ou modifiez la méthode `loadCahierCharges()` dans `database.py` pour changer le répertoire.

---

## Exécution manuelle (CLI)

```bash
python evaluateFunction.py \
    "[\"Test1.pdf\",\"Test2.pdf\"]" \
    "[\"poste1\",\"poste2\"]" \
    "[\"user1\",\"user2\"]" \
    "[\"POS\",\"FCT\"]" \
    "[\"false\",\"true\"]"
```

| Argument             | Description                                 |
|----------------------|---------------------------------------------|
| 1 – cahiers          | Liste de fichiers à analyser                |
| 2 – poste_ids        | Identifiants des postes                     |
| 3 – userids          | Identifiants utilisateurs                   |
| 4 – types_fonction   | POS, FCT, POSREQ, FCTREQ, COL               |
| 5 – lexicale_or_sem  | true = lexicale, false = sémantique         |

---

## Lancer l’API et l’interface Web

```bash
# Interface Web (formulaire .NET)
cd VbWebForm
dotnet run

# API FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
```

> Une soumission via formulaire déclenche automatiquement `evaluateFunction.py` avec les bons paramètres.

---

## Dockerisation

Le projet est compatible Docker pour éviter les installations locales.

- **Dockerfile** : instructions de construction de l’image
- **.dockerignore** : ignore les fichiers non nécessaires

### Build de l’image

```bash
docker build -t evaluationautomatique .
```

### Exécution

```bash
docker run -it --rm \
    -v "$(pwd)/clients:/EvalAutomatique/clients" \
    --add-host SPARK08:192.168.1.114 \
    evaluationautomatique \
    "[\"Test.pdf\"]" "[\"poste\"]" "[\"userid\"]" "[\"FCT\"]" "[\"true\"]"
```

> Adaptez les arguments aux fichiers, postes et paramètres que vous testez.

---

## Auteur

Issame Jekkam  
📧 jekkamissame@gmail.com