"""Расстояния и оценки центров локаций по обучающим взаимодействиям."""

import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """Расстояние в километрах; отсутствующие/некорректные координаты дают NaN."""
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (lat1, lon1, lat2, lon2)]
    )
    valid = (np.isfinite(lat1) & np.isfinite(lon1) & np.isfinite(lat2) & np.isfinite(lon2)
             & (np.abs(lat1) <= 90) & (np.abs(lat2) <= 90)
             & (np.abs(lon1) <= 180) & (np.abs(lon2) <= 180))
    # Подставляем нули только для расчёта; в результате невалидные пары станут NaN.
    lat1, lon1, lat2, lon2 = [np.radians(np.where(valid, x, 0))
                            for x in (lat1, lon1, lat2, lon2)]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return np.where(valid, 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))), np.nan)


def location_centers(train: pd.DataFrame) -> pd.DataFrame:
    """Медианы координат по item_location_id; fallback по search_location_id.

    Передаются только train-взаимодействия. Для fallback каждое объявление
    учитывается один раз на локацию запроса. Возвращаются center_lat/lon,
    число объявлений center_items и источник center_source; индекс — location_id.
    Последняя запись item_id задаёт его координаты внутри каждой группы.
    """
    columns = ["search_location_id", "item_id", "item_location_id", "item_latitude", "item_longitude"]
    data = train[columns].copy()
    for column in ("item_latitude", "item_longitude"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    def aggregate(group, dedup, source):
        part = data.drop_duplicates(dedup, keep="last")
        part = part.loc[part.item_latitude.between(-90, 90) & part.item_longitude.between(-180, 180)]
        centers = part.groupby(group).agg(
            center_lat=("item_latitude", "median"), center_lon=("item_longitude", "median"),
            center_items=("item_id", "size"),
        )
        centers.index.name = "location_id"
        centers["center_source"] = source
        return centers

    # Fallback заполняет только локации без центра; два источника не смешиваются.
    primary = aggregate("item_location_id", ["item_id"], "item_location")
    fallback = aggregate("search_location_id", ["search_location_id", "item_id"], "search_location")
    return pd.concat([primary, fallback.loc[~fallback.index.isin(primary.index)]]).sort_index()
