# House-Price-Prediction
This is my submission to the Kaggle Advanced Regression Competition : House Price Prediction. 

# House Price Prediction — Kaggle Top 12%

Predicting residential property sale prices using the 
[Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Final score:** 0.122 RMSE (log-scale) — Rank 469 / 4,106

---

## Approach

This was built iteratively over multiple submissions, not in one shot. The focus was on getting preprocessing right before tuning models.

**Preprocessing**
- Missing values handled column-by-column based on what the absence actually means (e.g. no garage → "None", not unknown)
- Ordinal features manually mapped to preserve order
- Nominal features one-hot encoded with unseen-category handling

**Feature engineering**
- `TotalLivArea`, `HouseAge`, `YearsSinceRemodel`, `QualityXArea` and 10+ others built from domain reasoning
- Out-of-fold target encoding for `Neighborhood` to capture price signal without leaking the target

**Modelling**
- XGBoost + LightGBM tuned with Optuna (100 trials each)
- Weighted ensemble blend, weights by inverse CV RMSE
- 5-fold cross-validation throughout

**What didn't work**
- Ridge regression in the ensemble consistently hurt Kaggle score
- Target encoding low-cardinality columns (MSZoning, BldgType) improved CV but worsened Kaggle — signs of overfitting
- Several interaction features added noise and were removed

---

## Results

| Submission | Kaggle RMSE | Notes |
|---|---|---|
| v1 | 0.12420 | Optuna XGBoost, no pipeline |
| v2 | 0.12538 | Full pipeline, 3-model ensemble |
| v3a | 0.12469 | XGB 60% + LGB 40% |
| v4 | 0.12424 | Stripped features |
| v5 | **0.12200** | + Neighborhood target encoding |

---

## Repo structure
```
├── notebook.ipynb    # full pipeline — EDA through submission
├── requirements.txt
└── data/             # not tracked — download from Kaggle
```

## Setup
```bash
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the 
[Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) 
and place them in the `data/` folder.

---

## Key learning

The biggest jump came from understanding *why* things worked — not just running models. 
OOF target encoding, pipeline-based preprocessing, and iterative feature removal all came from diagnosing specific failure modes, not from following a template.
