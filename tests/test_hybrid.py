"""Union, геопереранжирование и повторное использование готовых выдач."""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.retrievers.hybrid import HybridRetriever, RESULT_COLUMNS, haversine


def result(rows):
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame({
            "item_id": ["a", "b", "c", "d"],
            "item_location_id": [1, 1, 2, None],
            "item_latitude": [0, 0, 0, 95],
            "item_longitude": [10, 1, 0, 0],
        })
        self.queries = pd.DataFrame({
            "search_location_id": [1, None, 1],
            "center_lat": [0, None, 0],
            "center_lon": [0, None, 0],
        }, index=[8, 8, 2])
        self.outputs = {
            "bm25": result([(0, "a", 1, 99), (0, "b", 2, 8), (0, "c", 3, 1), (1, "d", 1, 4)]),
            "dense": result([(0, "b", 1, .7), (0, "d", 2, .2)]),
        }

    def test_union_rrf_and_top_n_per_source(self):
        model = HybridRetriever(n_candidates=2).fit(self.items)
        features = model.combine(self.queries, self.outputs)
        self.assertEqual(set(features.loc[features.query_index == 0, "item_id"]), {"a", "b", "d"})
        self.assertFalse(features.duplicated(["query_index", "item_id"]).any())
        top = model.rerank(features, k=10)
        self.assertEqual(top.item_id.tolist(), ["b", "a", "d", "d"])
        self.assertEqual(top["rank"].tolist(), [1, 2, 3, 1])
        self.assertEqual(top.query_index.tolist(), [0, 0, 0, 1])
        self.assertAlmostEqual(top.score.iloc[0], 1)
        # Несопоставимые исходные scores не меняют RRF.
        changed = {name: out.assign(score=out.score * -100) for name, out in self.outputs.items()}
        pd.testing.assert_frame_equal(features, model.combine(self.queries, changed))

    def test_geo_uses_full_union_before_final_cutoff(self):
        model = HybridRetriever(n_candidates=3, distance_only=True).fit(self.items)
        features = model.combine(self.queries, self.outputs)
        top = model.rerank(features, k=1)
        # c — третий у BM25 и отсутствует у dense, но находится ближе всех.
        self.assertEqual(top.item_id.tolist(), ["c", "d"])
        self.assertTrue(np.isfinite(top.score).all())
        model.set_params(n_candidates=1)
        smaller = model.rerank(model.combine(self.queries, self.outputs), k=1)
        self.assertEqual(smaller.item_id.tolist(), ["b", "d"])

    def test_missing_geo_is_neutral_and_missing_locations_do_not_match(self):
        model = HybridRetriever(location_weight=.2, distance_weight=.2).fit(self.items)
        features = model.combine(self.queries, self.outputs)
        missing = features.item_id.eq("d")
        self.assertTrue(features.loc[missing, "distance_km"].isna().all())
        self.assertFalse(features.loc[missing, "same_location"].any())
        top = model.rerank(features, k=10).set_index(["query_index", "item_id"])
        self.assertEqual(top.loc[(1, "d"), "score"], 1)
        # Для a есть и совпадение локации, и валидное расстояние.
        a = features.loc[features.item_id.eq("a")].iloc[0]
        self.assertAlmostEqual(top.loc[(0, "a"), "score"], a.text_score + .2 + .2 * np.exp(-a.distance_km / 100))

    def test_ready_models_are_not_refitted_and_cache_matches_retrieve(self):
        models = {name: Mock(retrieve=Mock(return_value=out)) for name, out in self.outputs.items()}
        model = HybridRetriever(models, n_candidates=3, location_weight=.2).fit(self.items)
        live = model.retrieve(self.queries, k=2)
        cached = model.rerank(model.combine(self.queries, self.outputs), k=2)
        pd.testing.assert_frame_equal(live, cached)
        for child in models.values():
            child.fit.assert_not_called()
            child.retrieve.assert_called_once_with(self.queries, k=3)

    def test_batch_positions_and_ties_are_stable(self):
        model = HybridRetriever(distance_only=True).fit(self.items)
        full = model.rerank(model.combine(self.queries, self.outputs), k=5)
        pieces = []
        for i in range(len(self.queries)):
            outputs = {name: out.loc[out.query_index == i].assign(query_index=0) for name, out in self.outputs.items()}
            piece = model.rerank(model.combine(self.queries.iloc[[i]], outputs), k=5)
            if not piece.empty:
                piece["query_index"] += i
                pieces.append(piece)
        pd.testing.assert_frame_equal(full, pd.concat(pieces, ignore_index=True))

    def test_empty_results_and_invalid_metadata(self):
        model = HybridRetriever().fit(self.items)
        self.assertTrue(model.rerank(model.combine(self.queries, {"empty": result([])}), k=50).empty)
        self.assertTrue(model.combine(self.queries.iloc[:0], {"empty": result([])}).empty)
        with self.assertRaisesRegex(ValueError, "absent"):
            model.combine(self.queries, {"bad": result([(0, "unknown", 1, 1)])})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            model.combine(self.queries, {"bad": result([(0, "a", 1, 1), (0, "a", 2, .5)])})
        with self.assertRaisesRegex(ValueError, "position"):
            model.combine(self.queries, {"bad": result([(3, "a", 1, 1)])})
        with self.assertRaisesRegex(ValueError, "unique"):
            model.fit(pd.concat([self.items, self.items]))

    def test_haversine_known_distances_and_bad_coordinates(self):
        distances = haversine([0, 0, 0, np.nan, 0], [0, 0, 0, 0, np.inf],
                              [0, 0, 100, 0, 0], [0, 90, 0, 0, 0])
        np.testing.assert_allclose(distances[:2], [0, 6371 * np.pi / 2])
        self.assertTrue(np.isnan(distances[2:]).all())


if __name__ == "__main__":
    unittest.main()
