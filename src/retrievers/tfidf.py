"""TF-IDF-поиск по словам или символьным n-граммам."""

from typing import Self
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from src.retriever import Retriever
from src.retrievers.bm25 import RussianTokenizer


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower().replace("ё", "е")


class TfidfRetriever(Retriever):
    """Косинусное сходство TF-IDF-векторов с L2-нормировкой.

    analyzer='word' использует RussianTokenizer, 'char' и 'char_wb' —
    нормализацию NFC, нижний регистр и ё→е без лемматизации.
    Матрицы разреженные; запросы обрабатываются батчами.
    Выдаются только положительные scores. При равенстве scores первым
    идёт объявление, раньше расположенное в корпусе при fit.
    """

    def __init__(
        self,
        *,
        item_text_column: str = "item_title_raw",
        query_text_column: str = "search_query",
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 1),
        lemmatize: bool = False,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        sublinear_tf: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.item_text_column = item_text_column
        self.query_text_column = query_text_column
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.lemmatize = lemmatize
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.batch_size = batch_size

    def fit(self, items: pd.DataFrame) -> Self:
        missing = {"item_id", self.item_text_column}.difference(items.columns)
        if missing:
            raise ValueError(f"Missing item columns: {sorted(missing)}")
        if items.empty:
            raise ValueError("Cannot index an empty item corpus.")
        if items["item_id"].isna().any() or items["item_id"].duplicated().any():
            raise ValueError("item_id must be unique and non-null.")
        if self.analyzer not in {"word", "char", "char_wb"}:
            raise ValueError("analyzer must be 'word', 'char' or 'char_wb'.")
        if self.lemmatize and self.analyzer != "word":
            raise ValueError("lemmatize is only supported for analyzer='word'.")

        tokenizer = (
            RussianTokenizer(lemmatize=self.lemmatize).tokenize
            if self.analyzer == "word" else None
        )
        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            tokenizer=tokenizer,
            preprocessor=_normalize if self.analyzer != "word" else None,
            token_pattern=None,
            lowercase=False,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(items[self.item_text_column].fillna(""))
        self.vectorizer_ = vectorizer
        self.item_vectors_ = matrix.tocsr()
        self.item_ids_ = items["item_id"].to_numpy(copy=True)
        return self

    def retrieve(self, queries: pd.DataFrame, *, k: int) -> pd.DataFrame:
        check_is_fitted(self, ["vectorizer_", "item_vectors_", "item_ids_"])
        if self.query_text_column not in queries.columns:
            raise ValueError(f"Missing query column: {self.query_text_column}")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        rows = []
        texts = queries[self.query_text_column].fillna("")
        for start in range(0, len(texts), self.batch_size):
            vectors = self.vectorizer_.transform(texts.iloc[start:start + self.batch_size])
            scores = (vectors @ self.item_vectors_.T).tocsr()
            for local_i in range(scores.shape[0]):
                left, right = scores.indptr[local_i:local_i + 2]
                doc_ids = scores.indices[left:right]
                values = scores.data[left:right]
                valid = np.isfinite(values) & (values > 0)
                doc_ids, values = doc_ids[valid], values[valid]
                if len(values) > k:
                    cutoff = np.partition(values, -k)[-k]
                    eligible = np.flatnonzero(values >= cutoff)
                else:
                    eligible = np.arange(len(values))
                # Сортируем только претендентов на top-k; при равенстве
                # score первым идёт документ с меньшим индексом корпуса.
                top = eligible[np.lexsort((doc_ids[eligible], -values[eligible]))[:k]]
                rows.extend(
                    (start + local_i, self.item_ids_[doc_ids[j]], rank, float(values[j]))
                    for rank, j in enumerate(top, start=1)
                )
        return pd.DataFrame.from_records(rows, columns=["query_index", "item_id", "rank", "score"])
