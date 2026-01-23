# T-DEV-810_msc2027_Zoiberg2.0
Projet Zoiberg Epitech 


# 🧠 ZOIDBERG 2.0 — Computer Aided Diagnosis

Projet de **machine learning / deep learning** visant à détecter la **pneumonie à partir d’images de radiographies pulmonaires (X-ray)**.  
Projet réalisé dans le cadre du module **T-DEV-810 (EPITECH)**.

---

## 🎯 Objectif du projet

À partir de plusieurs datasets d’images X-ray, le projet vise à :
- entraîner et comparer plusieurs modèles (baseline ML → CNN → transfer learning)
- mettre en place une méthodologie rigoureuse (train / val / test, cross-validation)
- évaluer les performances avec des métriques adaptées
- produire des résultats clairs, interprétables et reproductibles
- livrer un notebook technique ainsi qu’un document de synthèse (PDF)

---

## 🗂️ Architecture du projet

zoidberg/
├── data/
│   ├── train/
│   │   ├── malade/
│   │   └── non_malade/
│   ├── val/
│   │   ├── malade/
│   │   └── non_malade/
│   └── test/
│       ├── malade/
│       └── non_malade/
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_baseline_logistic.ipynb
│   ├── 03_cnn_scratch.ipynb
│   └── 04_transfer_learning.ipynb
│
├── src/
│   ├── init.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── logistic_baseline.pkl
│   ├── cnn_scratch.pt
│   └── transfer_learning.pt
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── environment.yml
├── README.md
└── .gitignore


---

## 📂 Description des dossiers

### `data/`
Contient les images X-ray organisées par :
- **split** : `train / val / test`
- **classe** : `malade / non_malade`

> ⚠️ Les données brutes ne doivent pas être versionnées si elles sont trop volumineuses.

---

### `notebooks/`
Notebooks Jupyter utilisés pour :
1. l’exploration des données
2. la mise en place d’une baseline ML (régression logistique)
3. l’entraînement d’un CNN from scratch
4. l’utilisation du transfer learning

Les notebooks suivent une **progression logique** du plus simple au plus avancé.

---

### `src/`
Code Python réutilisable :
- `data_loader.py` : chargement des images
- `preprocessing.py` : normalisation, resize, data augmentation
- `train.py` : entraînement des modèles
- `evaluate.py` : calcul des métriques, ROC, matrices de confusion

---

### `models/`
Modèles entraînés et sauvegardés :
- baseline ML
- CNN from scratch
- CNN en transfer learning

---

### `results/`
Résultats exploitables pour la synthèse finale :
- figures (ROC curves, confusion matrices, etc.)
- métriques sauvegardées (CSV / JSON)

---

## ⚙️ Installation de l’environnement

### Prérequis
- Miniconda
- Git configuré (SSH recommandé)

### Création de l’environnement
```bash
conda env create -f environment.yml
conda activate zoidberg-dl

python -m ipykernel install --user --name zoidberg-dl --display-name "zoidberg-dl"
jupyter notebook