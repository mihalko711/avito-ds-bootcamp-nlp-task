"""Общий интерфейс поиска объявлений по запросам."""

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


class Retriever(BaseEstimator, ABC):
    """Ищет объявления в корпусе и возвращает top-k для каждого запроса.

    Параметры метода задаются в __init__; обученное состояние хранится в
    атрибутах с суффиксом _. Это позволяет использовать get_params/set_params
    и clone из scikit-learn. Метрики вычисляются отдельно от retriever.

    fit получает корпус уникальных объявлений с item_id и полями, нужными
    реализации. Повторный fit заменяет индекс. Метки взаимодействий для
    базовых методов поиска не требуются.

    retrieve получает таблицу запросов с полями, нужными реализации.
    query_id не требуется. Результат — pandas.DataFrame со столбцами:

        query_index, item_id, rank, score

    query_index — позиция строки во входной таблице (от 0), независимо
    от её pandas index. Повторяющиеся тексты остаются отдельными запросами.
    Для каждого query_index допускается от 0 до k строк. Пары (query_index,
    item_id) уникальны; rank начинается с 1 и идёт без пропусков внутри
    запроса. Порядок строк соответствует порядку запросов, а внутри
    запроса — возрастанию rank. score — конечное число, чем выше, тем
    лучше; по возрастанию rank score не увеличивается. При равных score
    реализация задаёт стабильный порядок.

    Если совпадений нет, строк для этого запроса нет. При подсчёте метрик
    нужно учитывать все входные запросы, включая отсутствующие в выдаче.
    Для объединения результатов модели получают одну таблицу запросов
    в одинаковом порядке. Между отдельными вызовами query_index не является
    постоянным идентификатором; внешний код при необходимости связывает
    позиции с ID бенчмарка или другими ключами.

    Шкала score может отличаться у разных retriever. Перед взвешенным
    сложением их нужно приводить к сопоставимому масштабу; объединение
    по рангам не требует общей шкалы. Ансамбль тоже может реализовать
    этот интерфейс и искать по объединению выдач базовых retriever.
    """

    @abstractmethod
    def fit(self, items: pd.DataFrame) -> Self:
        """Построить индекс объявлений и вернуть self."""
        ...

    @abstractmethod
    def retrieve(self, queries: pd.DataFrame, *, k: int) -> pd.DataFrame:
        """Вернуть до k найденных объявлений для каждого запроса."""
        ...

    def score_candidates(self, queries: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
        """Посчитать скоры заданных пар без отбора top-k (опциональный метод).

        candidates содержит query_index (позицию в queries) и item_id.
        Возвращается массив в порядке строк candidates, включая повторы.
        NaN означает отсутствие сигнала: например, нет вектора объявления
        или запрос пуст. Нулевой BM25 при отсутствии совпадений — валидный скор.
        Статистики и векторы берутся из общего индекса, новый fit не выполняется.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support candidate scoring.")

    @staticmethod
    def _candidate_query_indices(queries: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
        """Проверить query_index кандидатов и вернуть позиции целыми числами."""
        if missing := {"query_index", "item_id"}.difference(candidates.columns):
            raise ValueError(f"Missing candidate columns: {sorted(missing)}")
        if candidates[["query_index", "item_id"]].isna().any().any():
            raise ValueError("Candidate query_index and item_id must be non-null.")
        positions = candidates.query_index
        if not (positions.between(0, len(queries) - 1) & positions.mod(1).eq(0)).all():
            raise ValueError("Candidate query_index must be a position in queries.")
        return positions.to_numpy(dtype=int)
