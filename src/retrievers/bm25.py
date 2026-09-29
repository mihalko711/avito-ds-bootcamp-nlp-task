"""BM25-поиск по тексту объявлений с токенизацией русского текста."""

from functools import lru_cache
import re
import unicodedata
from typing import Self

import bm25s
from bm25s.tokenization import Tokenizer
import numpy as np
import pandas as pd
from pymorphy3 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize
from sklearn.utils.validation import check_is_fitted

from src.retriever import Retriever


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILLIC = re.compile(r"^[а-я]+$")
_RESULT_COLUMNS = ["query_index", "item_id", "rank", "score"]


class RussianTokenizer:
    """Одинаково обрабатывает тексты объявлений и запросов.

    NFC, нижний регистр и ё→е применяются до разбиения razdel. Знаки
    пунктуации отделяются, а буквы и цифры сохраняются. Лемматизация
    кириллических слов опциональна; марки и модельные номера не меняются.
    Стоп-слова не удаляются.
    """

    def __init__(self, *, lemmatize: bool = False) -> None:
        """Создать токенизатор; морфологический анализатор нужен только для лемм."""
        self.lemmatize = lemmatize
        self._morph = MorphAnalyzer() if lemmatize else None

    @lru_cache(maxsize=100_000)
    def _lemma(self, word: str) -> str:
        """Вернуть частотную нормальную форму слова с ограниченным кешем."""
        assert self._morph is not None
        return self._morph.parse(word)[0].normal_form.replace("ё", "е")

    def tokenize(self, text: str) -> list[str]:
        """Нормализовать строку и вернуть слова, сохраняя числа и марки."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string; use an empty string for missing text.")
        normalized = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
        words = (
            word
            for token in razdel_tokenize(normalized)
            for word in _WORD.findall(token.text)
        )
        if self._morph is None:
            return list(words)
        return [self._lemma(word) if _CYRILLIC.fullmatch(word) else word for word in words]


class BM25Retriever(Retriever):
    """BM25S-индекс одного текстового поля объявлений.

    По умолчанию индексирует item_title_raw и ищет по search_query.
    Объявления с пустым текстом остаются в корпусе, но не выдаются.
    Нулевые scores от bm25s отбрасываются: библиотека может вернуть
    произвольные документы при отсутствии совпадений.
    """

    def __init__(
        self,
        *,
        item_text_column: str = "item_title_raw",
        query_text_column: str = "search_query",
        lemmatize: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
        batch_size: int = 256,
    ) -> None:
        """Задать текстовые поля, BM25-параметры и размер батча."""
        self.item_text_column = item_text_column
        self.query_text_column = query_text_column
        self.lemmatize = lemmatize
        self.k1 = k1
        self.b = b
        self.batch_size = batch_size

    def fit(self, items: pd.DataFrame) -> Self:
        """Построить BM25S-индекс по уникальным item_id из указанного поля."""
        required = {"item_id", self.item_text_column}
        missing = required.difference(items.columns)
        if missing:
            raise ValueError(f"Missing item columns: {sorted(missing)}")
        if items.empty:
            raise ValueError("Cannot index an empty item corpus.")
        if items["item_id"].isna().any() or items["item_id"].duplicated().any():
            raise ValueError("item_id must be unique and non-null.")
        if not np.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive.")
        if not np.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be finite and in [0, 1].")

        tokenizer = Tokenizer(
            splitter=RussianTokenizer(lemmatize=self.lemmatize).tokenize,
            lower=False,
            stopwords=[],
            stemmer=None,
        )
        texts = items[self.item_text_column].fillna("").tolist()
        tokens = tokenizer.tokenize(
            texts, update_vocab=True, return_as="tuple",
            allow_empty=False, show_progress=False,
        )
        if not tokens.vocab:
            raise ValueError("Item corpus contains no indexable tokens.")
        index = bm25s.BM25(k1=self.k1, b=self.b)
        index.index(tokens, show_progress=False)

        self.tokenizer_ = tokenizer
        self.index_ = index
        self.item_ids_ = items["item_id"].to_numpy(copy=True)
        return self

    def retrieve(self, queries: pd.DataFrame, *, k: int) -> pd.DataFrame:
        """Вернуть до k совпавших объявлений; пустые токенизированные запросы пропустить."""
        check_is_fitted(self, ["index_", "item_ids_", "tokenizer_"])
        required = {self.query_text_column}
        missing = required.difference(queries.columns)
        if missing:
            raise ValueError(f"Missing query columns: {sorted(missing)}")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        rows: list[tuple[int, object, int, float]] = []
        top_k = min(k, len(self.item_ids_))
        texts = queries[self.query_text_column].fillna("").tolist()
        for start in range(0, len(queries), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            tokens = self.tokenizer_.tokenize(
                batch_texts, update_vocab=False, allow_empty=False, show_progress=False,
            )
            active = [(i, token_list) for i, token_list in enumerate(tokens) if token_list]
            if not active:
                continue
            documents, scores = self.index_.retrieve(
                [token_list for _, token_list in active],
                k=top_k,
                show_progress=False,
            )
            for (local_i, _), doc_indices, doc_scores in zip(active, documents, scores):
                rank = 1
                for doc_i, score in zip(doc_indices, doc_scores):
                    if not np.isfinite(score) or score <= 0:
                        continue
                    rows.append((start + local_i, self.item_ids_[doc_i], rank, float(score)))
                    rank += 1

        return pd.DataFrame.from_records(rows, columns=_RESULT_COLUMNS)

    def score_candidates(self, queries: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
        """BM25 общего корпуса для заданных пар; нулевые совпадения сохраняются."""
        check_is_fitted(self, ["index_", "item_ids_", "tokenizer_"])
        if self.query_text_column not in queries:
            raise ValueError(f"Missing query column: {self.query_text_column}")
        query_indices = self._candidate_query_indices(queries, candidates)
        positions = pd.Index(self.item_ids_).get_indexer(candidates.item_id)
        scores = np.full(len(candidates), np.nan)
        for query_i in np.unique(query_indices):
            rows = np.flatnonzero((query_indices == query_i) & (positions >= 0))
            if not len(rows):
                continue
            text = queries[self.query_text_column].iloc[query_i]
            tokens = self.tokenizer_.tokenize(
                ["" if pd.isna(text) else text], update_vocab=False,
                allow_empty=False, show_progress=False,
            )[0]
            # BM25S суммирует postings по общему индексу; сохраняем только
            # скоры запрошенных объявлений, без глобального top-k.
            scores[rows] = self.index_.get_scores_from_ids(tokens)[positions[rows]] if tokens else 0
        return scores
