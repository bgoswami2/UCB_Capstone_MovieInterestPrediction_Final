"""
Data loading and integration helpers for the Movie Interest Prediction project.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict

import pandas as pd


ML_RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
ML_MOVIE_COLS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]
ML_USER_COLS = ["user_id", "age", "gender", "occupation", "zip_code"]


def _resolve_dataset_path(path: str | Path) -> Path:
    """
    Convert a possibly relative dataset path into an absolute path relative to
    the project's main directory.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    project_dir = Path(__file__).resolve().parents[1]
    return (project_dir / candidate).resolve()


def load_movielens_dataset(base_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load the MovieLens 100K dataset components.

    Returns a dictionary with keys: ratings, movies, users.
    """
    base = Path(base_path)
    ratings = pd.read_csv(
        base / "u.data",
        sep="\t",
        names=ML_RATINGS_COLS,
        header=None,
        engine="python",
    )
    movies = pd.read_csv(
        base / "u.item",
        sep="|",
        names=ML_MOVIE_COLS,
        header=None,
        encoding="latin-1",
    )
    users = pd.read_csv(
        base / "u.user",
        sep="|",
        names=ML_USER_COLS,
        header=None,
        encoding="latin-1",
    )
    return {"ratings": ratings, "movies": movies, "users": users}


def load_tmdb_dataset(base_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Load TMDB 5000 movie metadata and credits."""
    base = Path(base_path)
    movies = pd.read_csv(base / "tmdb_5000_movies.csv")
    credits = pd.read_csv(base / "tmdb_5000_credits.csv")
    return {"movies": movies, "credits": credits}


def _normalize_title(title: str | float) -> str:
    if not isinstance(title, str):
        return ""
    cleaned = re.sub(r"\([^)]*\)", " ", title)  # drop parenthetical year
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", cleaned).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _safe_literal_eval(value: str | float):
    if not isinstance(value, str) or not value:
        return []
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []


def _extract_genre_names(raw_value: str | float) -> list[str]:
    return [
        entry.get("name", "").strip()
        for entry in _safe_literal_eval(raw_value)
        if isinstance(entry, dict) and entry.get("name")
    ]


def _extract_top_cast(raw_value: str | float, top_n: int = 5) -> list[str]:
    cast_entries = [
        entry.get("name", "").strip()
        for entry in _safe_literal_eval(raw_value)
        if isinstance(entry, dict) and entry.get("name")
    ]
    return cast_entries[:top_n]


def prepare_movie_metadata(
    ml_movies: pd.DataFrame, tmdb_movies: pd.DataFrame, tmdb_credits: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine MovieLens movie information with TMDB metadata.

    Returns a movie dataframe indexed by MovieLens movie_id. The dataframe
    includes TMDB identifiers, genres, runtime, budget, overview, and top cast.
    """
    ml = ml_movies.copy()
    tmdb = tmdb_movies.copy()
    credits = tmdb_credits.copy()

    # Parse release years and normalized titles
    ml["release_year"] = (
        pd.to_datetime(ml["release_date"], errors="coerce").dt.year
    )
    ml["title_clean"] = ml["title"].apply(_normalize_title)
    ml["merge_key"] = ml.apply(
        lambda row: f"{row.title_clean}_{int(row.release_year)}"
        if pd.notna(row.release_year)
        else row.title_clean,
        axis=1,
    )

    tmdb["release_year"] = pd.to_datetime(
        tmdb["release_date"], errors="coerce"
    ).dt.year
    tmdb["title_clean"] = tmdb["title"].apply(_normalize_title)
    tmdb["merge_key"] = tmdb.apply(
        lambda row: f"{row.title_clean}_{int(row.release_year)}"
        if pd.notna(row.release_year)
        else row.title_clean,
        axis=1,
    )

    # Prioritize the most popular TMDB record when duplicates exist
    tmdb_sorted = (
        tmdb.sort_values(["merge_key", "popularity"], ascending=[True, False])
        .drop_duplicates("merge_key")
        .copy()
    )

    credits = credits.rename(columns={"title": "tmdb_title"})
    tmdb_enriched = tmdb_sorted.merge(
        credits[["movie_id", "tmdb_title", "cast", "crew"]],
        left_on="id",
        right_on="movie_id",
        how="left",
    )

    tmdb_enriched["genres_list"] = tmdb_enriched["genres"].apply(
        _extract_genre_names
    )
    tmdb_enriched["top_cast"] = tmdb_enriched["cast"].apply(
        _extract_top_cast
    )

    # Merge MovieLens movies with TMDB metadata via merge_key
    merged = ml.merge(
        tmdb_enriched,
        on="merge_key",
        how="left",
        suffixes=("", "_tmdb"),
    )

    merged = merged.rename(
        columns={
            "movie_id": "ml_movie_id",
            "movie_id_tmdb": "tmdb_movie_id",
            "id": "tmdb_id",
            "budget": "tmdb_budget",
            "genres": "tmdb_genres_raw",
            "overview": "tmdb_overview",
            "runtime": "tmdb_runtime",
            "release_date_tmdb": "tmdb_release_date",
            "vote_average": "tmdb_vote_average",
            "vote_count": "tmdb_vote_count",
        }
    )

    # Ensure we keep original MovieLens movie_id naming
    merged = merged.rename(columns={"ml_movie_id": "movie_id"})

    important_cols = [
        "movie_id",
        "title",
        "release_date",
        "imdb_url",
        "genres_list",
        "top_cast",
        "tmdb_movie_id",
        "tmdb_budget",
        "tmdb_runtime",
        "tmdb_overview",
        "tmdb_release_date",
        "tmdb_vote_average",
        "tmdb_vote_count",
        "popularity",
        "title_clean",
        "release_year",
    ]
    return merged[important_cols]


def build_enriched_ratings(
    ml_base_path: str | Path, tmdb_base_path: str | Path
) -> pd.DataFrame:
    """
    Create a unified ratings dataset joined with user demographics and TMDB metadata.

    Parameters
    ----------
    ml_base_path : str | Path
        Path to the MovieLens dataset directory. Relative paths are resolved
        against the project's main directory.
    tmdb_base_path : str | Path
        Path to the TMDB dataset directory. Relative paths are resolved against
        the project's main directory.
    """
    ml_base = _resolve_dataset_path(ml_base_path)
    tmdb_base = _resolve_dataset_path(tmdb_base_path)

    ml = load_movielens_dataset(ml_base)
    tmdb = load_tmdb_dataset(tmdb_base)
    movie_metadata = prepare_movie_metadata(
        ml["movies"], tmdb["movies"], tmdb["credits"]
    )

    ratings = ml["ratings"].merge(
        movie_metadata, on="movie_id", how="left", validate="many_to_one"
    )
    users = ml["users"]
    enriched = ratings.merge(users, on="user_id", how="left", validate="many_to_one")

    return enriched



