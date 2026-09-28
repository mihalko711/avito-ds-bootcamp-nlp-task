"""Проверки точного поиска и сохранения без скачивания нейросетевых весов."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.retrievers.dense import DenseRetriever


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, key):
        return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision")))

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs["prompt"]))
        vectors = {"a": [1, 0], "b": [0, 1], "c": [-1, 0], "q": [0.8, 0.6]}
        return np.array([vectors[text] for text in texts], dtype=np.float32)


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.encoder = FakeEncoder()
        self.loader = patch.object(DenseRetriever, "_load_encoder", return_value=self.encoder)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.items = pd.DataFrame({"item_id": ["A", "blank", "B", "C"], "item_title_raw": ["a", None, "b", "c"]})
        self.queries = pd.DataFrame({"search_query": ["q", "", "q"]}, index=[9, 9, 2])

    def test_exact_search_matches_cosine_and_batching(self):
        model = clone(DenseRetriever(batch_size=1, show_progress=False)).fit(self.items)
        out = model.retrieve(self.queries, k=10)
        self.assertEqual(out.query_index.tolist(), [0, 0, 0, 2, 2, 2])
        self.assertEqual(out.item_id.tolist(), ["A", "B", "C"] * 2)
        self.assertEqual(out["rank"].tolist(), [1, 2, 3] * 2)
        np.testing.assert_allclose(out.score, [0.8, 0.6, -0.8] * 2, atol=1e-6)
        model.batch_size = 8
        pd.testing.assert_frame_equal(out, model.retrieve(self.queries, k=10))
        self.assertTrue(model.retrieve(self.queries.iloc[:0], k=1).empty)
        self.assertTrue(all(prompt == "" for _, prompt in self.encoder.calls))

    def test_giga_prompt_and_save_load(self):
        model = DenseRetriever(model="giga", show_progress=False).fit(self.items)
        out = model.retrieve(self.queries, k=2)
        self.assertEqual(self.encoder.calls[0][1], "")
        self.assertEqual(self.encoder.calls[-1][1], f"Instruct: {model.query_instruction}\nQuery: ")
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = DenseRetriever.load(directory, device="cpu")
            self.assertFalse(hasattr(restored, "encoder_"))
            self.assertEqual(restored.revision, "test-revision")
            pd.testing.assert_frame_equal(out, restored.retrieve(self.queries, k=2))
        model.model = "bge"
        with self.assertRaisesRegex(ValueError, "changed after fit"):
            model.retrieve(self.queries, k=2)

    def test_refit_replaces_corpus_and_rejects_bad_input(self):
        model = DenseRetriever(show_progress=False).fit(self.items)
        model.fit(self.items.iloc[[2]])
        self.assertEqual(model.retrieve(self.queries, k=10).item_id.tolist(), ["B", "B"])
        with self.assertRaisesRegex(ValueError, "unique"):
            model.fit(pd.concat([self.items, self.items]))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            model.fit(self.items.iloc[[1]])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            model.retrieve(self.queries, k=0)

    def test_candidate_scores_use_saved_item_vectors_in_pair_order(self):
        model = DenseRetriever(show_progress=False).fit(self.items)
        pairs = pd.DataFrame({"query_index": [2, 0, 0, 1, 0, 0],
                              "item_id": ["C", "B", "A", "A", "blank", "unknown"]}, index=[9] * 6)
        self.encoder.calls.clear()
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = DenseRetriever.load(directory, device="cpu")
            scores = restored.score_candidates(self.queries, pairs)
        np.testing.assert_allclose(scores[:3], [-.8, .6, .8], atol=1e-6)
        self.assertTrue(np.isnan(scores[3:]).all())
        self.assertEqual(self.encoder.calls[0][0], ["q", "q"])
        self.assertEqual(len(restored.score_candidates(self.queries, pairs.iloc[:0])), 0)
        with self.assertRaisesRegex(ValueError, "position"):
            restored.score_candidates(self.queries, pairs.assign(query_index=99))
        restored.max_length += 1
        with self.assertRaisesRegex(ValueError, "changed after fit"):
            restored.score_candidates(self.queries, pairs)


if __name__ == "__main__":
    unittest.main()
