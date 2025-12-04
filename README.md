# Movie Interest Prediction - Capstone Final Submission

This repository contains the final technical deliverable for the Movie Interest Prediction capstone project. The goal is to help a streaming provider anticipate how strongly each user will rate unseen titles so that marketing teams can prioritize outreach and recommendation experiments that grow weekly active users.

## Project (sounce files) Contents
- [`capstone_final_modeling.ipynb`](capstone_final_modeling.ipynb) – end-to-end analysis notebook covering data intake, feature engineering, modeling, evaluation, and business-facing findings.
- `src/data.py` – helpers for loading MovieLens 100K and joining it with TMDB metadata.
- `src/features.py` – reusable feature engineering pipeline (numeric scaling, genre/user preference encodings, optional TF-IDF text vectors).
- `src/modeling.py` – modeling utilities (stratified splits, evaluation helpers, KNN and Random Forest trainers).

> The notebooks expect the raw MovieLens and TMDB CSV files to live under `./Data/` relative to this folder (`./Data/ml-100k`, `./Data/TMDB-5000`). Adjust `DATA_DIR`, `ML_PATH`, and `TMDB_PATH` inside the notebook if you keep them elsewhere.

## Project Summary (non-technical)
1. **Business challenge** – Identify which catalog titles each subscriber will enjoy so the marketing team can send fewer but higher impact campaigns.
2. **Data** – Historical ratings from MovieLens 100K enriched with TMDB genres, cast, runtime, and popularity plus user demographics.
3. **Method** – Consolidate the historical data, engineer leakage-safe user/movie features, and evaluate three model families (cosine KNN, Random Forest with grid search, and a tuned neural network) with hold-out testing and warm vs cold-start slices.
4. **Result** – All models beat a naive global-average benchmark. The Random Forest currently offers the lowest overall RMSE, while the neural network narrows the gap for under-observed titles.
5. **Impact** – Deploying the best model would cut RMSE roughly 6% versus the baseline, translating into more confident targeting for marketing experiments.

## Technical Approach
### 1. Data preparation
- Integrate MovieLens ratings, movies, and users with TMDB overview metadata using normalized title-year keys.
- Impute missing values conservatively (median runtime, budget/vote log transforms) and derive timestamps as pandas datetimes.
- Split data with the `user_stratified_split` helper so every user is represented in both train and test sets without leakage.

### 2. Feature engineering
- Multi-label binarization of movie genres plus rating-weighted user genre preference vectors.
- Numeric features (user averages/counts, runtime, popularity, vote stats, release year, genre match score) scaled via `StandardScaler`.
- Optional TF-IDF vectors over TMDB overviews (enabled in the final notebook with 300 terms and `min_df=5`).
- Sparse matrices assembled with SciPy for compatibility with both tree-based and neural models.

### 3. Modeling & evaluation
- **Cosine KNN baseline** for collaborative similarity.
- **Random Forest regressor** tuned with a 3-fold `GridSearchCV` (depth, estimators, min samples) and feature importance reporting.
- **Neural network** built with `keras_tuner.Hyperband`, early stopping, and dropout regularization.
- Metrics: RMSE and MAE on overall, warm (movies with history), and cold-start segments plus kernel density plots of prediction distributions.

## Results and Findings
| Model | Segment | RMSE | MAE |
| --- | --- | --- | --- |
| KNN | Overall | 1.123 | 0.883 |
| KNN | Cold-start | 1.273 | 1.004 |
| Random Forest | Overall | **1.063** | **0.847** |
| Random Forest | Cold-start | 1.201 | 0.973 |
| Neural Network | Overall | 1.078 | 0.860 |
| Neural Network | Cold-start | 1.230 | 0.996 |

- Random Forest currently provides the tightest error bounds overall and for warm catalog items while remaining interpretable through feature importances (user averages, release year, and genre preferences dominate).
- The tuned neural network nearly matches Random Forest on warm items and keeps cold-start RMSE within ~3% of the tree model, showing promise once more textual or content features are added.
- Cosine KNN, while weaker, is fast and useful as a sanity check for the more complex learners.

### Recommendations
1. **Production path** – Operationalize the Random Forest pipeline for immediate gains, using the `FeatureBuilder` artifacts to keep training/serving parity.
2. **Cold-start improvements** – Expand textual features (higher TF-IDF cap or transformer embeddings) and experiment with hybrid bandit exploration to further reduce cold-start error.
3. **Experimentation** – A/B test marketing journeys triggered by the model’s top-decile interest scores to estimate incremental uplift.

### Next Steps
- Incorporate additional streaming engagement signals (watch time, completion) once available.
- Explore matrix factorization with bias terms as another baseline.
- Extend evaluation to top-N ranking metrics (nDCG, hit rate) that align with recommendation UX KPIs.

## Reproducibility Checklist
1. **Environment** – Python 3.10+, install dependencies via  
   `pip install pandas numpy matplotlib seaborn scikit-learn scipy tensorflow keras-tuner plotly`.
2. **Data** – Download MovieLens 100K and TMDB 5000 datasets, unzip under `./Data/` as noted above.
3. **Run notebook** – Launch Jupyter Lab/Notebook, open `capstone_final_modeling.ipynb`, and run cells sequentially. Hardware with appropriate RAM is recommended for TF-IDF + neural training.
4. **Optional scripts** – Import helpers from `src/` when experimenting in separate notebooks or Python modules.
