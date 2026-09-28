"""Геопул, точный скоринг кандидатов, кортеж и нормированная сумма."""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.geo import location_centers
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.geo_first import GeoFirstRetriever


class GeoFirstTests(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame({
            "item_id": ["b", "a", "c", "d", "bad"],
            "item_latitude": [0, 0, 0, 0, 95], "item_longitude": [0, 0, 1, 2, 0],
        })
        self.queries = pd.DataFrame({"center_lat": [0, None, 0], "center_lon": [0, 0, 0]}, index=[8, 8, 2])

    def test_nearest_pool_ties_prefixes_and_unknown_centers(self):
        model = GeoFirstRetriever(n_candidates=4).fit(self.items)
        large = model.candidates(self.queries)
        self.assertEqual(large.item_id.tolist(), ["a", "b", "c", "d"] * 2)
        self.assertEqual(large.query_index.tolist(), [0] * 4 + [2] * 4)
        np.testing.assert_allclose(large.distance_km.iloc[:4], [0, 0, np.pi * 6371 / 180, np.pi * 6371 / 90])
        model.set_params(n_candidates=1)
        small = model.candidates(self.queries)
        pd.testing.assert_frame_equal(small, large.loc[large.geo_rank <= 1].reset_index(drop=True))
        self.assertTrue(model.candidates(self.queries.iloc[[1]]).empty)

    def test_tuple_uses_later_scores_then_distance_only_for_ties(self):
        model = GeoFirstRetriever(n_candidates=4, fusion="tuple", score_order=("first", "second")).fit(self.items)
        pool = model.candidates(self.queries.iloc[:1])
        pool["first"] = [2, 2, 1, 2]
        pool["second"] = [1, 1, 1000, 2]
        result = model.rerank(pool, k=4)
        self.assertEqual(result.item_id.tolist(), ["d", "a", "b", "c"])
        self.assertTrue(result.score.is_monotonic_decreasing)
        # Даже большой второй скор c не компенсирует меньший первый.
        self.assertEqual(result["rank"].tolist(), [1, 2, 3, 4])

    def test_sum_normalizes_each_query_and_handles_constant_missing_negative(self):
        model = GeoFirstRetriever(n_candidates=4, fusion="sum", score_order=("bm25", "dense", "empty", "constant")).fit(self.items)
        pool = model.candidates(self.queries.iloc[:1])
        pool["bm25"] = [100, 90, 0, 0]
        pool["dense"] = [-1, 0, 1, np.nan]
        pool["empty"] = np.nan
        pool["constant"] = 20
        ranked = model.rerank(pool, k=4)
        self.assertEqual(ranked.item_id.tolist(), ["b", "a", "c", "d"])
        np.testing.assert_allclose(ranked.score, [1.4, 1, 1, 0])
        # Изменение масштаба одного источника не меняет сумму после min–max.
        pd.testing.assert_frame_equal(ranked, model.rerank(pool.assign(bm25=pool.bm25 * 1000), k=4))
        model.set_params(n_candidates=2)
        smaller = model.rerank(pool, k=2)
        self.assertEqual(smaller.item_id.tolist(), ["a", "b"])
        np.testing.assert_allclose(smaller.score, [1, 1])

    def test_ready_models_score_every_geo_candidate_without_global_retrieve(self):
        child = Mock()
        child.score_candidates.side_effect = lambda q, pairs: pairs.item_id.map({"a": 0, "b": 2, "c": 3}).to_numpy()
        model = GeoFirstRetriever({"text": child}, n_candidates=3, fusion="tuple").fit(self.items)
        result = model.retrieve(self.queries.iloc[:1], k=1)
        self.assertEqual(result.item_id.tolist(), ["c"])
        child.fit.assert_not_called()
        child.retrieve.assert_not_called()
        self.assertEqual(len(child.score_candidates.call_args.args[1]), 3)

    def test_centers_prefer_item_location_and_deduplicate_query_item_fallback(self):
        train = pd.DataFrame({
            "search_location_id": [100, 100, 100, 100, 200],
            "item_location_id": [1, 1, 1, 2, 200],
            "item_id": ["a", "a", "a", "b", "invalid"],
            "item_latitude": [10, 10, 10, 30, 100], "item_longitude": [20, 20, 20, 40, 0],
        })
        centers = location_centers(train)
        self.assertEqual(centers.loc[100, "center_lat"], 20)
        self.assertEqual(centers.loc[100, "center_items"], 2)
        self.assertEqual(centers.loc[100, "center_source"], "search_location")
        self.assertEqual(centers.loc[1, "center_lat"], 10)
        self.assertNotIn(200, centers.index)


class BM25CandidateTests(unittest.TestCase):
    def test_pair_scores_match_global_retrieval_including_zero_matches(self):
        items = pd.DataFrame({"item_id": ["a", "b", "c", "blank"],
                              "item_title_raw": ["красный телефон", "синий телефон", "стол", None]})
        queries = pd.DataFrame({"search_query": ["телефон", "", "несуществующееслово"]}, index=[9, 9, 8])
        model = BM25Retriever().fit(items)
        pairs = pd.DataFrame({"query_index": [0, 0, 0, 0, 0, 1, 2, 0],
                              "item_id": ["b", "a", "c", "blank", "unknown", "a", "b", "b"]})
        scores = model.score_candidates(queries, pairs)
        reference = model.retrieve(queries.iloc[:1], k=4).set_index("item_id").score
        np.testing.assert_allclose(scores[:2], reference.loc[["b", "a"]])
        np.testing.assert_allclose(scores[[2, 3, 5, 6]], 0)
        self.assertTrue(np.isnan(scores[4]))
        self.assertEqual(scores[7], scores[0])
        self.assertEqual(len(model.score_candidates(queries, pairs.iloc[:0])), 0)


if __name__ == "__main__":
    unittest.main()
