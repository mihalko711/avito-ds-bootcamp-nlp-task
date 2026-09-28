"""Воспроизводимое разбиение взаимодействий по текстам запросов."""

from collections.abc import Iterable
import unicodedata

import numpy as np
from numpy.typing import NDArray


def normalize_query_group(text: str) -> str:
    """Фиксированная нормализация для сплита, независимая от скореров."""
    if not isinstance(text, str):
        raise ValueError("Query must be a string; handle missing values explicitly.")
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    return " ".join(text.split())


def split_queries(
    queries: Iterable[str],
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> NDArray[np.str_]:
    """Вернуть метки train/val/test в исходном порядке строк.

    Принимает последовательность текстов, в том числе pandas/Polars Series.
    Все варианты одного нормализованного запроса попадают в одну часть.
    Доли относятся к числу уникальных групп, а не к числу взаимодействий:
    размеры val и test округляются вверх, остаток составляет train.
    Нулевая доля отключает соответствующую часть.

    Для одинакового набора групп, seed и долей результат не зависит от
    порядка строк и количества повторов. При изменении набора групп
    разбиение может измениться: для воспроизводимости фиксируйте также
    версию данных или сохраняйте соответствие групп и частей.

    Пустой ввод, нестроковые запросы, недопустимые доли и недостаточное
    число групп вызывают ValueError. Пустые строки образуют общую группу.
    Корпус объявлений эта функция не разделяет и пары не удаляет.
    """
    if not (
        np.isfinite(val_size)
        and np.isfinite(test_size)
        and 0 <= val_size < 1
        and 0 <= test_size < 1
        and val_size + test_size < 1
    ):
        raise ValueError("Split fractions must be nonnegative and sum to less than 1.")

    normalized = [normalize_query_group(query) for query in queries]
    groups = np.array(sorted(set(normalized)), dtype=str)
    if not len(groups):
        raise ValueError("Cannot split an empty query collection.")

    n_val = int(np.ceil(len(groups) * val_size))
    n_test = int(np.ceil(len(groups) * test_size))
    if n_val + n_test >= len(groups):
        raise ValueError("Too few query groups for the requested split fractions.")

    shuffled = np.random.default_rng(seed).permutation(groups)
    assignments = dict.fromkeys(shuffled, "train")
    assignments.update(dict.fromkeys(shuffled[:n_test], "test"))
    assignments.update(dict.fromkeys(shuffled[n_test : n_test + n_val], "val"))
    return np.array([assignments[group] for group in normalized], dtype="<U5")
