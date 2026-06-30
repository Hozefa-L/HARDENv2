"""TF-IDF features from opcode text corpus.

Fits TfidfVectorizer on training split only, transforms all splits.
Produces a parquet file with fp_runtime_unified + split + tfidf_* columns.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OPCODE_CORPUS_PATH = PROJECT_ROOT / "data" / "features" / "main_benchmark" / "opcode_text_corpus.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "main_benchmark" / "tfidf_features.parquet"

DEFAULT_MAX_FEATURES = 500
DEFAULT_NGRAM_RANGE = (1, 2)


def build_tfidf_features(
    corpus_path: Path = DEFAULT_OPCODE_CORPUS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: tuple = DEFAULT_NGRAM_RANGE,
    sublinear_tf: bool = True,
    train_split: str = "train",
) -> pd.DataFrame:
    """Build TF-IDF features from opcode text corpus.

    Fits on training split only, transforms all contracts.

    Returns:
        DataFrame with columns: fp_runtime_unified, split, tfidf_0 .. tfidf_{n-1}
    """
    logger.info("Loading opcode text corpus from %s", corpus_path)
    corpus = pd.read_parquet(corpus_path)

    required_cols = {"fp_runtime_unified", "split", "opcode_text"}
    missing = required_cols - set(corpus.columns)
    if missing:
        raise ValueError(f"Opcode corpus missing columns: {missing}")

    corpus["fp_runtime_unified"] = corpus["fp_runtime_unified"].astype(str).str.strip()
    corpus["opcode_text"] = corpus["opcode_text"].fillna("").astype(str)

    train_mask = corpus["split"] == train_split
    train_count = int(train_mask.sum())
    if train_count == 0:
        raise ValueError(f"No rows with split='{train_split}' in corpus.")
    logger.info("Fitting TF-IDF on %d training samples (max_features=%d, ngram_range=%s)",
                train_count, max_features, ngram_range)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )
    vectorizer.fit(corpus.loc[train_mask, "opcode_text"])

    logger.info("Transforming all %d samples", len(corpus))
    tfidf_matrix = vectorizer.transform(corpus["opcode_text"])

    feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=corpus.index,
        columns=feature_names,
    )

    result = pd.concat([
        corpus[["fp_runtime_unified", "split"]].reset_index(drop=True),
        tfidf_df.reset_index(drop=True),
    ], axis=1)

    actual_features = tfidf_matrix.shape[1]
    vocab_size = len(vectorizer.vocabulary_)
    logger.info("TF-IDF features: %d dimensions, vocab size: %d", actual_features, vocab_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved TF-IDF features to %s (%d rows, %d columns)",
                output_path, len(result), len(result.columns))

    return result


def get_vectorizer_vocab(
    corpus_path: Path = DEFAULT_OPCODE_CORPUS_PATH,
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: tuple = DEFAULT_NGRAM_RANGE,
    sublinear_tf: bool = True,
    train_split: str = "train",
) -> dict:
    """Fit and return the TF-IDF vocabulary (for inspection/debugging)."""
    corpus = pd.read_parquet(corpus_path)
    corpus["opcode_text"] = corpus["opcode_text"].fillna("").astype(str)
    train_mask = corpus["split"] == train_split

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )
    vectorizer.fit(corpus.loc[train_mask, "opcode_text"])
    return vectorizer.vocabulary_


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build TF-IDF opcode features")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_OPCODE_CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    args = parser.parse_args()

    build_tfidf_features(
        corpus_path=args.corpus,
        output_path=args.output,
        max_features=args.max_features,
    )
