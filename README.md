# CrashOutcome-Classifier

Predict the severity of road accidents in France using machine learning.

## Context

This project originates from a private Kaggle competition organized for students of the **DSIA-4301A** program (Data Science & AI) at **ESIEE Paris**, in collaboration with the **Laboratoire des Mécanismes d'Accidents (LMA)** at **Université Gustave Eiffel**. The competition ran from March 20 to May 12, 2024.

I originally competed and finished **1st place** with an AUC of **0.85376**. This repository was created post-competition in February 2026 to revisit the problem and try to improve that score.

**Author:** Lucien Laumont (laumontlucien@gmail.com)

## Problem Statement

Given a set of features describing a road accident (characteristics, location, vehicles, users), the objective is to predict the **probability that the accident is severe**.

A binary target variable `GRAVE` is defined as:
- **1** if at least one involved user was killed or hospitalized (`grav` = 2 or 3)
- **0** otherwise

Performance is measured by the **AUC** (Area Under the ROC Curve).

## Data

The data comes from the French national road accident database (BAAC). It is split into four tables:

| File | Description |
|------|-------------|
| `characteristics.csv` | General accident attributes (date, time, lighting, weather, etc.) |
| `locations.csv` | Road and location details (road type, infrastructure, surface condition, etc.) |
| `vehicles.csv` | Vehicle information (category, maneuver, obstacle hit, etc.) |
| `users.csv` | User-level data (severity, age, gender, safety equipment, etc.) |

### Structure

```
data/
├── train/              # Training data (2010-2022), one folder per year
│   ├── BAAC-Annee-2010/
│   │   ├── characteristics.csv
│   │   ├── locations.csv
│   │   ├── users.csv
│   │   └── vehicles.csv
│   ├── BAAC-Annee-2011/
│   │   └── ...
│   └── ...
├── test/               # Test set (no target variable)
│   ├── characteristics.csv
│   ├── locations.csv
│   ├── users.csv
│   └── vehicles.csv
doc/
├── sample_submission.csv
└── *.pdf               # Official ONISR data documentation
```

### Submission Format

```csv
Num_Acc,GRAVE
201200049538,0.7432
201200004221,0.2315
201200002457,0.5239
```

## Approach

The competition allows any method: feature engineering, table merging, clustering, PCA, non-linear models, neural networks, etc. The key is to be rigorous in parameter selection and systematically evaluate performance.
