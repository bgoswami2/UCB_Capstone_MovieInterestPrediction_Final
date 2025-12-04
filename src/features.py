"""
Feature engineering utilities for the Movie Interest Prediction project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


def _normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.lower()).strip("_")
    return cleaned or "unknown"


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [value]
    return []


@dataclass
class FeatureArtifacts:
    numeric_features: List[str] = field(default_factory=list)
    genre_features: List[str] = field(default_factory=list)
    user_pref_features: List[str] = field(default_factory=list)
    text_features: List[str] = field(default_factory=list)
    vectorizer: Optional[TfidfVectorizer] = None
    scaler: Optional[StandardScaler] = None
    mlb: Optional[MultiLabelBinarizer] = None

    @property
    def all_features(self) -> List[str]:
        return (
            list(self.numeric_features)
            + list(self.genre_features)
            + list(self.user_pref_features)
            + list(self.text_features)
        )


class FeatureBuilder:
    """Engineer model-ready features while preventing data leakage."""

    def __init__(
        self,
        include_text: bool = False,
        max_text_features: int = 500,
        min_text_df: int = 3,
    ) -> None:
        self.include_text = include_text
        self.max_text_features = max_text_features
        self.min_text_df = min_text_df
        self.artifacts = FeatureArtifacts()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        if df.empty:
            raise ValueError("Cannot fit FeatureBuilder on an empty dataframe.")

        work = df.copy()
        self.global_avg_rating_ = work["rating"].mean()
        self.user_avg_rating_ = work.groupby("user_id")["rating"].mean()
        self.user_rating_count_ = work.groupby("user_id")["rating"].count()

        genres_series = work["genres_list"].apply(_ensure_list)
        self.artifacts.mlb = MultiLabelBinarizer()
        genre_matrix = self.artifacts.mlb.fit_transform(genres_series)
        genre_classes = list(self.artifacts.mlb.classes_)
        self.genre_feature_names_ = [
            f"genre_{_normalize_label(name)}" for name in genre_classes
        ]

        # User preference vectors weighted by ratings
        user_ids = work["user_id"].values
        rating_weights = work["rating"].values.reshape(-1, 1)
        weighted_genre = genre_matrix * rating_weights
        weighted_df = pd.DataFrame(
            weighted_genre, columns=genre_classes
        ).assign(user_id=user_ids)
        agg = weighted_df.groupby("user_id")[genre_classes].sum()
        self.user_genre_pref_ = agg.div(agg.sum(axis=1), axis=0).fillna(0.0)
        self.user_pref_feature_names_ = [
            f"user_pref_{_normalize_label(name)}" for name in genre_classes
        ]

        # TF-IDF on overviews if requested
        if self.include_text:
            self.artifacts.vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=self.max_text_features,
                min_df=self.min_text_df,
            )
            self.artifacts.vectorizer.fit(work["tmdb_overview"].fillna(""))
            self.text_feature_names_ = [
                f"text_{feat}"
                for feat in self.artifacts.vectorizer.get_feature_names_out()
            ]
        else:
            self.text_feature_names_ = []

        # Numeric features to scale
        work["user_avg_rating"] = work["user_id"].map(self.user_avg_rating_).fillna(
            self.global_avg_rating_
        )
        work["user_rating_count"] = work["user_id"].map(self.user_rating_count_).fillna(0)
        work["tmdb_budget_log"] = np.log1p(work["tmdb_budget"].fillna(0))
        work["tmdb_vote_count_log"] = np.log1p(work["tmdb_vote_count"].fillna(0))
        work["tmdb_runtime"] = work["tmdb_runtime"].fillna(work["tmdb_runtime"].median())
        work["tmdb_vote_average"] = work["tmdb_vote_average"].fillna(
            work["tmdb_vote_average"].median()
        )
        work["popularity"] = work["popularity"].fillna(work["popularity"].median())
        work["release_year"] = work["release_year"].fillna(
            work["release_year"].median()
        )

        user_pref = self.user_genre_pref_.reindex(work["user_id"]).fillna(0.0)
        match_score = (genre_matrix * user_pref.values).sum(axis=1)
        work["genre_match_score"] = match_score

        numeric_cols = [
            "user_avg_rating",
            "user_rating_count",
            "tmdb_runtime",
            "tmdb_budget_log",
            "tmdb_vote_average",
            "tmdb_vote_count_log",
            "popularity",
            "release_year",
            "genre_match_score",
        ]
        self.artifacts.numeric_features = numeric_cols

        self.artifacts.genre_features = self.genre_feature_names_
        self.artifacts.user_pref_features = self.user_pref_feature_names_
        self.artifacts.text_features = self.text_feature_names_

        self.artifacts.scaler = StandardScaler()
        self.artifacts.scaler.fit(self._build_numeric_matrix(work))

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fitted before calling transform.")

        work = df.copy()
        work["user_avg_rating"] = work["user_id"].map(self.user_avg_rating_).fillna(
            self.global_avg_rating_
        )
        work["user_rating_count"] = work["user_id"].map(self.user_rating_count_).fillna(0)

        work["tmdb_budget_log"] = np.log1p(work["tmdb_budget"].fillna(0))
        work["tmdb_vote_count_log"] = np.log1p(work["tmdb_vote_count"].fillna(0))
        work["tmdb_runtime"] = work["tmdb_runtime"].fillna(work["tmdb_runtime"].median())
        work["tmdb_vote_average"] = work["tmdb_vote_average"].fillna(
            work["tmdb_vote_average"].median()
        )
        work["popularity"] = work["popularity"].fillna(work["popularity"].median())
        work["release_year"] = work["release_year"].fillna(
            work["release_year"].median()
        )

        # Movie genre multi-hot
        genres_series = work["genres_list"].apply(_ensure_list)
        genre_matrix = self.artifacts.mlb.transform(genres_series)

        # User preference columns
        user_pref = self.user_genre_pref_.reindex(work["user_id"]).fillna(0.0)

        # Genre match score = dot product of multi-hot and user preference
        match_score = (genre_matrix * user_pref.values).sum(axis=1)
        work["genre_match_score"] = match_score

        numeric_matrix = self.artifacts.scaler.transform(
            self._build_numeric_matrix(work)
        )

        # Assemble feature matrices
        genre_feature_matrix = genre_matrix.astype(float)
        user_pref_matrix = user_pref.values.astype(float)

        components = [
            sparse.csr_matrix(numeric_matrix),
            sparse.csr_matrix(genre_feature_matrix),
            sparse.csr_matrix(user_pref_matrix),
        ]

        if self.include_text and self.artifacts.vectorizer is not None:
            text_matrix = self.artifacts.vectorizer.transform(
                work["tmdb_overview"].fillna("")
            )
            components.append(text_matrix)

        X = sparse.hstack(components).tocsr()
        return X

    def fit_transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return self.fit(df).transform(df)

    def _build_numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        numeric_df = df[
            [
                "user_avg_rating",
                "user_rating_count",
                "tmdb_runtime",
                "tmdb_budget_log",
                "tmdb_vote_average",
                "tmdb_vote_count_log",
                "popularity",
                "release_year",
                "genre_match_score",
            ]
        ]
        return numeric_df.to_numpy()

    def get_feature_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fitted before requesting feature names.")
        return self.artifacts.all_features


