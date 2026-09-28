"""Ближайшие N объявлений → скоры заданных пар → кортеж или нормированная сумма."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.utils.validation import check_is_fitted

from src.retriever import Retriever


RESULT_COLUMNS = ["query_index", "item_id", "rank", "score"]
POOL_COLUMNS = ["query_index", "item_id", "geo_rank", "distance_km"]


class GeoFirstRetriever(Retriever):
    """Предварительный отбор по расстоянию и текстовое ранжирование внутри пула.

    retrievers — готовые модели с score_candidates. fit строит только
    геоиндекс, не обучая текстовые модели. queries содержит center_lat/lon.
    Неизвестный центр или невалидные координаты объявления исключают его
    из геопоиска. Никакого неявного добавления глобальных текстовых кандидатов.

    fusion='tuple': убывающие сырые скоры в порядке score_order, затем
    возрастающее расстояние и item_id. Каждый следующий скор разрешает
    только полные ничьи предыдущих. NaN ниже любого известного скора.
    Выходной score=-rank отражает лексикографический порядок, а не релевантность.

    fusion='sum': min–max каждого скора внутри top-N одного запроса, затем
    сумма. Константный/отсутствующий сигнал даёт 0. weights задаёт необязательные
    неотрицательные веса; по умолчанию все равны 1. Нормировка пересчитывается
    при изменении N. fusion='distance' — контроль чистого геопоиска.

    Для кеша доступны candidates(queries), score_candidates(queries, pool)
    через вложенные модели и rerank(scored, k=50). scored содержит POOL_COLUMNS
    и колонки с именами моделей из score_order.
    """

    def __init__(self, retrievers: dict[str, Retriever] | None = None, *, n_candidates: int = 500,
                 fusion: str = "sum", score_order: tuple[str, ...] | None = None,
                 weights: dict[str, float] | None = None) -> None:
        self.retrievers = retrievers
        self.n_candidates = n_candidates
        self.fusion = fusion
        self.score_order = score_order
        self.weights = weights

    def _validate_params(self):
        if isinstance(self.n_candidates, bool) or not isinstance(self.n_candidates, int) or self.n_candidates < 1:
            raise ValueError("n_candidates must be a positive integer.")
        if self.fusion not in {"tuple", "sum", "distance"}:
            raise ValueError("fusion must be tuple, sum or distance.")

    def fit(self, items: pd.DataFrame) -> Self:
        """Индексировать только валидные координаты; текстовые индексы уже готовы."""
        self._validate_params()
        if items.item_id.isna().any() or items.item_id.duplicated().any():
            raise ValueError("item_id must be unique and non-null.")
        coords = items[["item_latitude", "item_longitude"]].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(coords).all(axis=1) & (np.abs(coords[:, 0]) <= 90) & (np.abs(coords[:, 1]) <= 180)
        if not valid.any():
            raise ValueError("No items with valid coordinates.")
        self.item_ids_ = items.item_id.to_numpy()[valid]
        self.tree_ = BallTree(np.radians(coords[valid]), metric="haversine")
        return self

    def candidates(self, queries: pd.DataFrame) -> pd.DataFrame:
        """Вернуть ближайшие n_candidates, стабильно разрешая ничьи по item_id."""
        check_is_fitted(self, ["tree_", "item_ids_"])
        self._validate_params()
        coords = queries[["center_lat", "center_lon"]].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(coords).all(axis=1) & (np.abs(coords[:, 0]) <= 90) & (np.abs(coords[:, 1]) <= 180)
        active = np.flatnonzero(valid)
        if not len(active):
            return pd.DataFrame(columns=POOL_COLUMNS)
        centers, inverse = np.unique(np.radians(coords[active]), axis=0, return_inverse=True)
        k = min(self.n_candidates, len(self.item_ids_))
        distances, _ = self.tree_.query(centers, k=k)
        # Включаем все ничьи на границе top-N: префиксы при разных N совпадают.
        neighbors, distances = self.tree_.query_radius(
            centers, r=distances[:, -1] + 1e-12, return_distance=True,
        )
        pools = []
        for indices, dist in zip(neighbors, distances):
            order = np.lexsort((self.item_ids_[indices], dist))[:k]
            pools.append((self.item_ids_[indices[order]], dist[order] * 6371))
        rows = []
        for query_i, center_i in zip(active, inverse):
            ids, dist = pools[center_i]
            rows.extend((query_i, item_id, rank, float(distance))
                        for rank, (item_id, distance) in enumerate(zip(ids, dist), 1))
        return pd.DataFrame.from_records(rows, columns=POOL_COLUMNS)

    def rerank(self, scored: pd.DataFrame, *, k: int) -> pd.DataFrame:
        """Выбрать top-k из заранее посчитанных скоров геопула."""
        self._validate_params()
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        out = scored.loc[scored.geo_rank <= self.n_candidates].copy()
        if out.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        names = list(self.score_order if self.score_order is not None else (self.retrievers or {}))
        if self.fusion != "distance" and (not names or len(set(names)) != len(names)):
            raise ValueError("Provide distinct model names in score_order or retrievers.")
        if self.fusion != "distance" and np.isinf(out[names].to_numpy(dtype=float)).any():
            raise ValueError("Model scores must be finite or NaN.")
        if self.fusion == "tuple":
            out = out.sort_values(["query_index", *names, "distance_km", "item_id"],
                                  ascending=[True, *([False] * len(names)), True, True], na_position="last")
        else:
            if self.fusion == "distance":
                out["score"] = -out.distance_km
            else:
                low = out.groupby("query_index")[names].transform("min")
                span = out.groupby("query_index")[names].transform("max") - low
                normalized = ((out[names] - low) / span.replace(0, np.nan)).fillna(0)
                weights = np.array([(self.weights or {}).get(name, 1.0) for name in names])
                if not np.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("Weights must be finite and nonnegative.")
                out["score"] = normalized.to_numpy() @ weights
            out = out.sort_values(["query_index", "score", "distance_km", "item_id"],
                                  ascending=[True, False, True, True])
        out = out.groupby("query_index", sort=False).head(k).copy()
        out["rank"] = out.groupby("query_index", sort=False).cumcount() + 1
        if self.fusion == "tuple":
            out["score"] = -out["rank"].astype(float)
        return out[RESULT_COLUMNS].reset_index(drop=True)

    def retrieve(self, queries: pd.DataFrame, *, k: int) -> pd.DataFrame:
        scored = self.candidates(queries)
        if self.fusion != "distance":
            if not self.retrievers:
                raise ValueError("Pass fitted retrievers, or use rerank with cached scores.")
            for name, model in self.retrievers.items():
                scored[name] = model.score_candidates(queries, scored)
        return self.rerank(scored, k=k)
