"""Объединение выдач готовых ретриверов и географическое переранжирование."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.retriever import Retriever
from src.geo import haversine


RESULT_COLUMNS = ["query_index", "item_id", "rank", "score"]
GEO_COLUMNS = ["item_location_id", "item_latitude", "item_longitude"]
FEATURE_COLUMNS = ["query_index", "item_id", "text_score", "same_location", "distance_km"]


class HybridRetriever(Retriever):
    """top-N каждого источника → union по item_id → RRF + гео → top-k.

    retrievers — словарь УЖЕ построенных ретриверов. fit сохраняет только
    геометаданные объявлений и не вызывает fit вложенных моделей: dense
    индексы можно загрузить через DenseRetriever.load без переиндексации.

    Запросы содержат search_location_id, center_lat и center_lon. Центры
    локаций готовит вызывающий код, например только по train. Неизвестный
    центр задаётся NaN. Это приближение локации, а не координаты пользователя.

    RRF = sum(1 / (rrf_constant + rank)); отсутствующий в выдаче документ
    получает от источника 0. RRF делится на максимум внутри запроса.
    score = text_score + location_weight * same_location
            + distance_weight * exp(-distance_km / distance_scale_km).
    Неизвестное расстояние и отсутствующая локация дают нулевой геобонус.

    distance_only=True сортирует по расстоянию; неизвестные расстояния
    идут последними. При равных итоговых scores порядок задают text_score
    и item_id. Если расстояний нет вообще, это даёт порядок RRF.

    Для экспериментов с кешем можно не передавать retrievers:
    combine(queries, results) строит таблицу признаков из сохранённых выдач,
    rerank(features, k=50) выбирает финальную выдачу. Повторный rerank после
    set_params меняет геовеса без поиска и повторного объединения.
    """

    def __init__(
        self,
        retrievers: dict[str, Retriever] | None = None,
        *,
        n_candidates: int = 500,
        rrf_constant: float = 60,
        location_weight: float = 0,
        distance_weight: float = 0,
        distance_scale_km: float = 100,
        distance_only: bool = False,
    ) -> None:
        self.retrievers = retrievers
        self.n_candidates = n_candidates
        self.rrf_constant = rrf_constant
        self.location_weight = location_weight
        self.distance_weight = distance_weight
        self.distance_scale_km = distance_scale_km
        self.distance_only = distance_only

    def _validate_params(self):
        if isinstance(self.n_candidates, bool) or not isinstance(self.n_candidates, int) or self.n_candidates < 1:
            raise ValueError("n_candidates must be a positive integer.")
        for name in ("rrf_constant", "location_weight", "distance_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.distance_scale_km) or self.distance_scale_km <= 0:
            raise ValueError("distance_scale_km must be finite and positive.")

    def fit(self, items: pd.DataFrame) -> Self:
        """Сохранить геометаданные; вложенные ретриверы должны быть готовы."""
        self._validate_params()
        required = ["item_id", *GEO_COLUMNS]
        missing = set(required).difference(items.columns)
        if missing:
            raise ValueError(f"Missing item columns: {sorted(missing)}")
        if items.item_id.isna().any() or items.item_id.duplicated().any():
            raise ValueError("item_id must be unique and non-null.")
        self.items_ = items[required].set_index("item_id").copy()
        return self

    def combine(self, queries: pd.DataFrame, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Объединить top-N из каждой выдачи; query_index — позиция в queries."""
        check_is_fitted(self, ["items_"])
        self._validate_params()
        required = {"search_location_id", "center_lat", "center_lon"}
        if missing := required.difference(queries.columns):
            raise ValueError(f"Missing query columns: {sorted(missing)}")
        if not results:
            raise ValueError("Provide at least one retrieval result.")
        parts = []
        for name, result in results.items():
            part = result.loc[result["rank"] <= self.n_candidates, ["query_index", "item_id", "rank"]].copy()
            if part.empty:
                continue
            if part[["query_index", "item_id", "rank"]].isna().any().any():
                raise ValueError(f"{name}: null query_index, item_id or rank.")
            if not (part.query_index.between(0, len(queries) - 1) & part.query_index.mod(1).eq(0)).all():
                raise ValueError(f"{name}: query_index is not a position in queries.")
            if not (part["rank"].ge(1) & part["rank"].mod(1).eq(0)).all():
                raise ValueError(f"{name}: rank must be a positive integer.")
            if part.duplicated(["query_index", "item_id"]).any():
                raise ValueError(f"{name}: duplicate query/item pairs.")
            part["text_score"] = 1 / (self.rrf_constant + part["rank"])
            parts.append(part.drop(columns="rank"))
        if not parts:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        combined = (pd.concat(parts, ignore_index=True)
                    .groupby(["query_index", "item_id"], sort=False, as_index=False)["text_score"].sum())
        combined["text_score"] /= combined.groupby("query_index")["text_score"].transform("max")
        positions = self.items_.index.get_indexer(combined.item_id)
        if (positions < 0).any():
            raise ValueError("Candidates contain item_id absent from the metadata corpus.")
        item = self.items_.iloc[positions].reset_index(drop=True)
        query = queries.iloc[combined.query_index.to_numpy(dtype=int)].reset_index(drop=True)
        combined["same_location"] = (item.item_location_id.eq(query.search_location_id)
                                     & item.item_location_id.notna() & query.search_location_id.notna()).fillna(False)
        coords = [series.to_numpy(dtype=float, na_value=np.nan) for series in
                  (query.center_lat, query.center_lon, item.item_latitude, item.item_longitude)]
        combined["distance_km"] = haversine(*coords)
        return combined[FEATURE_COLUMNS]

    def rerank(self, candidates: pd.DataFrame, *, k: int) -> pd.DataFrame:
        """Переранжировать результат combine; исходная таблица не изменяется."""
        self._validate_params()
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if candidates.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        out = candidates[["query_index", "item_id", "text_score"]].copy()
        distance = candidates.distance_km.to_numpy(dtype=float, na_value=np.nan)
        if self.distance_only:
            out["score"] = -np.where(np.isfinite(distance), distance, np.finfo(float).max)
        else:
            proximity = np.exp(-np.where(np.isfinite(distance), distance, np.inf) / self.distance_scale_km)
            out["score"] = (out.text_score + self.location_weight * candidates.same_location
                            + self.distance_weight * proximity)
        out = out.sort_values(["query_index", "score", "text_score", "item_id"],
                              ascending=[True, False, False, True], kind="stable")
        out = out.groupby("query_index", sort=False).head(k).copy()
        out["rank"] = out.groupby("query_index", sort=False).cumcount() + 1
        return out[RESULT_COLUMNS].reset_index(drop=True)

    def retrieve(self, queries: pd.DataFrame, *, k: int) -> pd.DataFrame:
        check_is_fitted(self, ["items_"])
        self._validate_params()
        if not self.retrievers:
            raise ValueError("Pass fitted retrievers, or use combine/rerank with cached results.")
        results = {name: model.retrieve(queries, k=self.n_candidates)
                   for name, model in self.retrievers.items()}
        return self.rerank(self.combine(queries, results), k=k)
