"""BGE-M3 и Giga-480M: биэнкодер и точный поиск FAISS по cosine similarity.

Установка зависимостей: uv sync --extra dense.
"""

import json
from pathlib import Path
from typing import Self

import faiss
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.retriever import Retriever


MODELS = {
    "bge": "BAAI/bge-m3",
    "giga": "ai-sage/Giga-Embeddings-instruct-480M-0826",
}
RESULT_COLUMNS = ["query_index", "item_id", "rank", "score"]


class DenseRetriever(Retriever):
    """fit кодирует корпус без обучения весов; retrieve возвращает top-k.

    Векторы нормируются и порциями добавляются в CPU IndexFlatIP. Полная
    матрица query×corpus не создаётся. device управляет только энкодером.
    dtype='auto' выбирает float16 на CUDA и float32 на CPU; индекс — float32.

    Для Giga подключается код из её HF-репозитория (trust_remote_code=True),
    а перед запросом добавляется query_instruction. BGE префиксов не требует.
    Тексты передаются штатному токенизатору без лемматизации и смены регистра.

    Пустые/null тексты объявлений не индексируются, пустые запросы не дают
    выдачи. Непустые запросы получают min(k, размер индекса) кандидатов,
    включая кандидатов с нулевой или отрицательной cosine similarity.

    save сохраняет FAISS-индекс с векторами, порядком item_id и настройками.
    load восстанавливает индекс без повторного кодирования корпуса; веса
    энкодера загружаются лениво при первом запросе. Для воспроизводимости
    можно задать HF commit в revision; определённая при fit ревизия также
    сохраняется, если доступна в конфигурации модели.
    """

    def __init__(
        self,
        *,
        model: str = "bge",
        item_text_column: str = "item_title_raw",
        query_text_column: str = "search_query",
        query_instruction: str = "Найди объявления, соответствующие поисковому запросу.",
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int = 32,
        max_length: int = 256,
        revision: str | None = None,
        show_progress: bool = True,
    ) -> None:
        self.model = model
        self.item_text_column = item_text_column
        self.query_text_column = query_text_column
        self.query_instruction = query_instruction
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.revision = revision
        self.show_progress = show_progress

    def _validate_params(self):
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {tuple(MODELS)}")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dtype.")
        for name in ("batch_size", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _texts(frame, column):
        if column not in frame.columns:
            raise ValueError(f"Missing text column: {column}")
        texts = frame[column].fillna("").tolist()
        if not all(isinstance(text, str) for text in texts):
            raise ValueError(f"{column} must contain strings or nulls.")
        return texts

    def _load_encoder(self):
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.dtype
        if dtype == "auto":
            dtype = "float16" if device.startswith("cuda") else "float32"
        encoder = SentenceTransformer(
            MODELS[self.model],
            device=device,
            revision=self.revision,
            trust_remote_code=self.model == "giga",
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
        )
        encoder.max_seq_length = self.max_length
        return encoder

    def _encode(self, texts, *, query):
        prompt = ""
        if query and self.model == "giga":
            prompt = f"Instruct: {self.query_instruction}\nQuery: "
        vectors = self.encoder_.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("Encoder returned invalid embeddings.")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Encoder returned a zero vector.")
        faiss.normalize_L2(vectors)
        return vectors

    def fit(self, items: pd.DataFrame) -> Self:
        from tqdm.auto import tqdm

        self._validate_params()
        if "item_id" not in items.columns:
            raise ValueError("Missing item_id column.")
        if items["item_id"].isna().any() or items["item_id"].duplicated().any():
            raise ValueError("item_id must be unique and non-null.")
        texts = self._texts(items, self.item_text_column)
        positions = [i for i, text in enumerate(texts) if text.strip()]
        if not positions:
            raise ValueError("Corpus contains no nonempty texts.")
        # Освобождаем прежний индекс/энкодер до нового fit и не оставляем
        # старую выдачу доступной при ошибке во время переиндексации.
        for name in ("index_", "item_ids_", "encoder_"):
            self.__dict__.pop(name, None)
        self.encoder_ = self._load_encoder()
        index = None
        for start in tqdm(range(0, len(positions), self.batch_size),
                          desc=f"Encode corpus ({self.model})", disable=not self.show_progress):
            batch_positions = positions[start : start + self.batch_size]
            vectors = self._encode([texts[i] for i in batch_positions], query=False)
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        self.index_ = index
        self.item_ids_ = items["item_id"].iloc[positions].to_numpy(copy=True)
        auto_model = getattr(self.encoder_[0], "auto_model", None)
        self.resolved_revision_ = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
        self.fitted_params_ = self.get_params(deep=False).copy()
        return self

    def retrieve(self, queries: pd.DataFrame, *, k: int) -> pd.DataFrame:
        check_is_fitted(self, ["index_", "item_ids_"])
        self._validate_params()
        # Эти настройки меняют пространство векторов: после их изменения
        # нужен новый fit. Размер батча и вывод прогресса менять можно.
        for name in ("model", "revision", "max_length", "query_instruction", "dtype"):
            if getattr(self, name) != self.fitted_params_[name]:
                raise ValueError(f"{name} changed after fit; rebuild the index.")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        texts = self._texts(queries, self.query_text_column)
        positions = [i for i, text in enumerate(texts) if text.strip()]
        if not positions:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        if not hasattr(self, "encoder_"):
            self.encoder_ = self._load_encoder()
        rows = []
        for start in range(0, len(positions), self.batch_size):
            batch_positions = positions[start : start + self.batch_size]
            vectors = self._encode([texts[i] for i in batch_positions], query=True)
            scores, indices = self.index_.search(vectors, min(k, self.index_.ntotal))
            for query_index, doc_indices, doc_scores in zip(batch_positions, indices, scores):
                rows.extend(
                    (query_index, self.item_ids_[doc_i], rank, float(score))
                    for rank, (doc_i, score) in enumerate(zip(doc_indices, doc_scores), 1)
                )
        return pd.DataFrame.from_records(rows, columns=RESULT_COLUMNS)

    def save(self, directory: str | Path) -> None:
        """Сохранить индекс (включая векторы) и настройки; без весов модели."""
        check_is_fitted(self, ["index_", "item_ids_"])
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        params = self.fitted_params_.copy()
        params["revision"] = self.resolved_revision_ or params["revision"]
        metadata = json.dumps(
            {"params": params, "item_ids": self.item_ids_.tolist()}, ensure_ascii=False,
        )
        faiss.write_index(self.index_, str(path / "index.faiss"))
        (path / "metadata.json").write_text(metadata, encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, *, device: str | None = None) -> Self:
        """Восстановить индекс; device=None автоматически выбирает устройство."""
        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        params = metadata["params"]
        params["device"] = device
        instance = cls(**params)
        instance._validate_params()
        instance.index_ = faiss.read_index(str(path / "index.faiss"))
        instance.item_ids_ = np.asarray(metadata["item_ids"])
        if len(instance.item_ids_) != instance.index_.ntotal:
            raise ValueError("Index and item_ids have different lengths.")
        instance.fitted_params_ = params.copy()
        instance.resolved_revision_ = params["revision"]
        return instance
