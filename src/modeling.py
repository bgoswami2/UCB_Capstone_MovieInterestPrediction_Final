"""
Model training helpers for the Movie Interest Prediction project.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def user_stratified_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_user_ratings: int = 5,
    min_test_items: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/test sets by holding out the most recent ratings per user.

    Ensures every user has at least `min_test_items` samples in test (where possible)
    while keeping at least one observation in train.
    """
    if "timestamp" not in df.columns:
        raise ValueError("Dataframe must contain a timestamp column for stratified split.")

    train_parts = []
    test_parts = []

    for _user_id, group in df.groupby("user_id"):
        sorted_group = group.sort_values("timestamp")
        n_ratings = len(sorted_group)
        if n_ratings < min_user_ratings:
            n_test = min_test_items if n_ratings > min_test_items else 1
        else:
            n_test = max(min_test_items, math.ceil(n_ratings * test_ratio))
        n_test = min(n_test, n_ratings - 1) if n_ratings > 1 else 0

        if n_test <= 0:
            train_parts.append(sorted_group)
            continue

        test_part = sorted_group.tail(n_test)
        train_part = sorted_group.iloc[: n_ratings - n_test]

        train_parts.append(train_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE and MAE metrics."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}


def cold_start_mask(
    df: pd.DataFrame,
    historical_counts: pd.Series,
    min_count: int = 5,
) -> np.ndarray:
    """
    Identify rows corresponding to movies with fewer than `min_count` historical ratings.
    """
    if "movie_id" not in df.columns:
        raise ValueError("Dataframe must include movie_id to compute cold start mask.")
    movie_counts = historical_counts.reindex(df["movie_id"]).fillna(0)
    return (movie_counts < min_count).to_numpy()


def fit_baseline_knn(
    X_train,
    y_train,
    n_neighbors: int = 25,
    metric: str = "cosine",
) -> KNeighborsRegressor:
    """Fit a simple KNN regressor."""
    model = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric, weights="distance")
    model.fit(X_train, y_train)
    return model


def fit_random_forest(
    X_train,
    y_train,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest regressor with sensible defaults."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def random_forest_feature_importance(
    model: RandomForestRegressor, feature_names
) -> pd.DataFrame:
    """Return a sorted dataframe of feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    )


