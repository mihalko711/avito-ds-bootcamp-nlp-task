"""Small end-to-end comparison and submission contract checks; no GPU required."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.retrieval_comparison import RetrievalComparison, write_answer


class ComparisonTests(unittest.TestCase):
    def test_submission_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'answer.csv'
            ids = ['0000000000000001', 'abcdef0123456789']
            queries = ['Aa00000000000001', 'aa00000000000001']
            frame = write_answer(queries, [ids, []], ids, path)
            self.assertEqual(frame.query_id.tolist(), queries)
            self.assertEqual(pd.read_csv(path, keep_default_na=False).answer.iloc[1], '')
            for invalid in [[ids * 2, []], [['ffffffffffffffff'], []], [[ids[1].upper()], []]]:
                with self.assertRaises(ValueError):
                    write_answer(queries, invalid, ids, path)
            with self.assertRaises(ValueError):
                write_answer(queries * 2, [[], [], [], []], ids, path)
            with self.assertRaises(ValueError):
                write_answer(queries, [ids], ids, path)

    def test_two_approaches_and_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(200):
                rows.append(dict(item_id=f'{i:016x}', search_query=f'телефон {i}',
                    search_location_id=1, search_is_delivery_search=0,
                    search_infm_params_text='цвет красный', search_category=114,
                    item_location_id=1, item_latitude=55.0+i/10000,
                    item_longitude=37.0, item_rating=float(i % 6),
                    item_rating_reviews_count=float(i), item_is_phone_hidden=bool(i % 2),
                    item_is_message_forbidden=bool(i % 3), item_title_raw=f'телефон {i}',
                    item_description_raw=f'красный телефон {i}', item_infm_params_text='цвет красный'))
            data = pl.DataFrame(rows)
            train_path = root/'train.parquet'
            data.write_parquet(train_path)
            experiment = RetrievalComparison(repo_path=Path.cwd(),data_path=train_path,
                dense_index_dir=None,dense_models=[],seed=42,sample_seed=43,n_val=3,n_test=4,
                ns=[50,100],ks=[10,50,100],select_k=50,query_batch_size=2,dense_batch_size=2,
                bm25_lemmatize=False,score_order=['bm25_title','bm25_description','bm25_infm'],
                quality_depth=100,quality_weights=[0.,.1],artifacts=root/'artifacts')
            experiment.prepare_data()
            experiment.prepare_cache()
            experiment.build()
            experiment.validate()
            result = experiment.test()
            self.assertEqual(len(result),12)
            self.assertTrue(result['Recall@50'].between(0,1).all())
            self.assertTrue((result['Recall@50'] <= result.pool_recall + 1e-12).all())
            # Positive quality shifts can cross top-K membership; rank IDs stay unique.
            base = pd.DataFrame({'query_index':[0]*100, 'item_id':experiment.items.item_id[:100],
                                 'rank':range(1,101)})
            reranked = experiment.late_rerank(base,'reviews',.1)
            self.assertEqual(set(reranked['rank']),set(range(1,101)))
            self.assertFalse(reranked.item_id.duplicated().any())
            self.assertEqual(experiment.late_rerank(base,'reviews',0).item_id.tolist(),base.item_id.tolist())
            item_path=root/'benchmark_items.parquet'
            data.select('item_id',*experiment.geo_cols,*experiment.quality_cols,*experiment.text_cols).write_parquet(item_path)
            query_path=root/'benchmark_queries.parquet'
            queries=data.select(experiment.query_cols).head(3).with_columns(pl.Series('query_id',[f'Query{i:011d}' for i in range(3)]))
            # Unknown location must remain a row even if geo-first returns no candidates.
            queries=queries.with_columns(pl.when(pl.col('query_id')=='Query00000000002').then(999).otherwise(pl.col('search_location_id')).alias('search_location_id'))
            queries.write_parquet(query_path)
            for approach in ['text_then_geo','geo_then_tuple']:
                answer=experiment.benchmark(items_path=item_path,queries_path=query_path,
                    index_dir=root/'benchmark_indexes',build_missing_indexes=False,
                    approach=approach,n=50,feature='reviews',weight=.1,output_path=root/f'{approach}.csv')
                self.assertEqual(answer.query_id.tolist(),queries['query_id'].to_list())
                self.assertEqual(list(answer.columns),['query_id','answer'])
                self.assertTrue(answer.answer.str.split().map(len).le(50).all())
                if approach=='geo_then_tuple': self.assertEqual(answer.answer.iloc[2],'')

    def test_notebook_code_compiles(self):
        notebook=json.loads(Path('notebooks/Kaggle_retrieval_comparison.ipynb').read_text())
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code': compile(''.join(cell['source']),f'cell_{i}','exec')


if __name__ == '__main__':
    unittest.main()
