"""Batched Kaggle comparison using existing retrievers; disk caches and paired evaluation."""
from pathlib import Path
import gc
import hashlib
import json
import re
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from src.split import split_queries
from src.geo import location_centers
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.geo_first import GeoFirstRetriever

class RetrievalComparison:
    """Orchestrate existing retrievers; tune on val, evaluate test, export benchmark."""

    def __init__(self, *, repo_path, data_path, dense_index_dir, dense_models, seed, sample_seed, n_val, n_test, ns, ks, select_k, query_batch_size, dense_batch_size, bm25_lemmatize, score_order, quality_depth, quality_weights, artifacts, dense_max_length=128, build_dense_indices=True):
        self.root = repo_path
        self.data_path = data_path
        self.dense_index_dir = dense_index_dir
        self.dense_models = dense_models
        self.seed = seed
        self.sample_seed = sample_seed
        self.n_val = n_val
        self.n_test = n_test
        self.ns = ns
        self.ks = ks
        self.select_k = select_k
        self.query_batch_size = query_batch_size
        self.dense_batch_size = dense_batch_size
        self.dense_max_length = dense_max_length
        self.build_dense_indices = build_dense_indices
        self.bm25_lemmatize = bm25_lemmatize
        self.score_order = score_order
        self.quality_depth = quality_depth
        self.quality_weights = quality_weights
        self.artifacts = artifacts
        self._setup()


    def _setup(self):
        if self.dense_models:
            import torch
            from src.retrievers.dense import DenseRetriever
            if not torch.cuda.is_available():
                raise RuntimeError('Включите Accelerator: GPU в Kaggle.')
        assert all((n >= self.select_k for n in self.ns)) and self.select_k in self.ks
        assert self.query_batch_size > 0 and self.quality_depth >= max(self.ks)
        self.query_cols = ['search_query', 'search_location_id', 'search_is_delivery_search', 'search_infm_params_text', 'search_category']
        self.geo_cols = ['item_location_id', 'item_latitude', 'item_longitude']
        self.quality_cols = ['item_rating', 'item_rating_reviews_count', 'item_is_phone_hidden', 'item_is_message_forbidden']
        self.text_cols = ['item_title_raw', 'item_description_raw', 'item_infm_params_text']
        required = set(self.query_cols + self.geo_cols + self.quality_cols + self.text_cols + ['item_id'])
        if self.data_path is None:
            paths = [self.root / 'data/train.parquet', *Path('/kaggle/input').rglob('train.parquet')]
            paths = list({p.resolve() for p in paths if p.is_file() and required.issubset(pl.read_parquet_schema(p))})
            if len(paths) != 1:
                raise ValueError(f'Укажите DATA_PATH: подходящих train.parquet — {len(paths)}.')
            self.data_path = paths[0]
        self.data_path = Path(self.data_path)
        if (missing := (required - set(pl.read_parquet_schema(self.data_path)))):
            raise ValueError(f'В train.parquet отсутствуют {missing}')
        if self.dense_models and self.dense_index_dir is None:
            roots = [p.parent.parent for p in Path('/kaggle/input').rglob('index.faiss')]
            roots = list({p.resolve() for p in roots if all(
                (p / name / 'index.faiss').is_file() and (p / name / 'metadata.json').is_file()
                for name in self.dense_models)})
            if len(roots) > 1:
                raise ValueError(f'Несколько папок с dense-индексами: {roots}. Укажите DENSE_INDEX_DIR.')
            if roots:
                self.dense_index_dir = roots[0]
            else:
                self.dense_index_dir = (Path('/kaggle/working/retrieval_comparison/dense_indexes')
                                        if Path('/kaggle/working').is_dir()
                                        else self.artifacts / 'dense_indexes')
        if self.dense_index_dir is not None:
            self.dense_index_dir = Path(self.dense_index_dir)
            self.dense_index_dir.mkdir(parents=True, exist_ok=True)
        self.dense_metadata = {}
        print('Код:', self.root, 'Данные:', self.data_path, 'Индексы:', self.dense_index_dir)

    def read_item_text(self, column):
        frame = pl.scan_parquet(self.data_path).select('item_id', column).unique(subset='item_id', keep='last', maintain_order=True).sort('item_id').collect()
        return pd.DataFrame(frame.to_dict(as_series=False))

    def prepare_data(self):
        from src.geo import location_centers
        self.query_cols = ['search_query', 'search_location_id', 'search_is_delivery_search', 'search_infm_params_text', 'search_category']
        self.geo_cols = ['item_location_id', 'item_latitude', 'item_longitude']
        data = pl.read_parquet(self.data_path, columns=self.query_cols + ['item_id'] + self.geo_cols + self.quality_cols)
        data = data.with_columns(pl.Series('split', split_queries(data['search_query'], seed=self.seed)), pl.col('item_latitude', 'item_longitude').cast(pl.Float64))
        self.items = pd.DataFrame(data.select('item_id', *self.geo_cols, *self.quality_cols).unique(subset='item_id', keep='last', maintain_order=True).sort('item_id').to_dict(as_series=False))
        train_items = data.filter(pl.col('split') == 'train').select('search_location_id', 'item_id', *self.geo_cols)
        self.centers = location_centers(pd.DataFrame(train_items.to_dict(as_series=False)))
        self.queries, self.targets = ({}, {})
        for split in ('val', 'test'):
            part = data.filter(pl.col('split') == split).group_by(self.query_cols).agg(pl.col('item_id').unique().alias('targets')).sort(self.query_cols)
            limit = self.n_val if split == 'val' else self.n_test
            if limit is not None:
                part = part.sample(n=min(limit, part.height), seed=self.sample_seed, shuffle=True)
            self.queries[split] = pd.DataFrame(part.select(self.query_cols).to_dict(as_series=False)).join(self.centers, on='search_location_id')
            self.targets[split] = [set(ids) for ids in part['targets'].to_list()]
            assert all(self.targets[split])
            assert set().union(*self.targets[split]).issubset(set(self.items.item_id))
            print(split, 'контекстов:', len(part), 'без центра:', self.queries[split].center_lat.isna().sum())
        display(data.group_by('split').len().sort('split'))
        train_ids = set(train_items['item_id'].to_list())
        train_quality = self.items.loc[self.items.item_id.isin(train_ids)]
        self.rating_mean = train_quality.item_rating.dropna().clip(0, 5).mean()
        self.rating_mean = float(self.rating_mean) if pd.notna(self.rating_mean) else 0.0
        scale = float(np.log1p(train_quality.item_rating_reviews_count.fillna(0).clip(lower=0).quantile(0.99)))
        self.review_scale = max(scale, 1.0) if np.isfinite(scale) else 1.0
        del data, train_items, part, train_quality, train_ids
        gc.collect()
        for split, q in self.queries.items():
            print(split, 'источник центра:')
            display(q.center_source.value_counts(dropna=False))
        self.prepare_quality()

    def prepare_dense_indices(self):
        """Build and cache each item index on the Kaggle GPU; reuse it on reruns."""
        if not self.dense_models:
            return
        from src.retrievers.dense import DenseRetriever
        import torch
        corpus = self.read_item_text('item_title_raw')
        expected = corpus.loc[corpus.item_title_raw.fillna('').str.strip().ne(''), 'item_id'].tolist()
        for name in self.dense_models:
            path = self.dense_index_dir / name
            if (path / 'index.faiss').is_file() and (path / 'metadata.json').is_file():
                model = DenseRetriever.load(path, device='cuda')
                if (model.model != name or model.item_text_column != 'item_title_raw'
                        or model.query_text_column != 'search_query' or model.max_length != self.dense_max_length):
                    raise ValueError(f'{path}: параметры индекса не совпадают; удалите этот индекс или задайте новый DENSE_INDEX_DIR.')
                if model.item_ids_.tolist() != expected:
                    raise ValueError(f'{path}: корпус отличается; задайте новый DENSE_INDEX_DIR.')
                print(name, '— индекс уже есть:', path)
            else:
                if not self.build_dense_indices:
                    raise FileNotFoundError(f'{path}: индекс отсутствует. Включите BUILD_DENSE_INDEXES для построения на Kaggle.')
                model = DenseRetriever(model=name, item_text_column='item_title_raw',
                    query_text_column='search_query', device='cuda', batch_size=self.dense_batch_size,
                    max_length=self.dense_max_length, show_progress=True)
                print(name, '— кодирую объявления на Kaggle GPU')
                model.fit(corpus)
                model.save(path)
                print(name, '— индекс сохранён:', path)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            raw = (path / 'metadata.json').read_bytes()
            stat = (path / 'index.faiss').stat()
            self.dense_metadata[name] = [hashlib.sha256(raw).hexdigest(), stat.st_size, stat.st_mtime_ns]
        del corpus, expected
        gc.collect()

    def prepare_quality(self):
        self.quality = pd.DataFrame(index=self.items.item_id)
        rating = self.items.item_rating.to_numpy(dtype=float)
        counts = self.items.item_rating_reviews_count.to_numpy(dtype=float)
        n = np.nan_to_num(np.maximum(counts, 0), nan=0)
        adjusted = (n * np.nan_to_num(np.clip(rating, 0, 5), nan=self.rating_mean) + 20 * self.rating_mean) / (n + 20)
        self.quality['rating'] = np.where(np.isfinite(rating), (adjusted - self.rating_mean) / 5, 0)
        self.quality['reviews'] = np.where(np.isfinite(counts), np.clip(np.log1p(n) / self.review_scale, 0, 1), 0)
        for name, col in [('phone', 'item_is_phone_hidden'), ('messages', 'item_is_message_forbidden')]:
            x = self.items[col].to_numpy(dtype=float)
            self.quality[name] = np.where(np.isfinite(x), 1 - 2 * np.clip(x, 0, 1), 0)
        self.quality['all'] = self.quality.mean(axis=1)

    def cache_path(self, kind, split, start, name='pool'):
        return self.cache / kind / name / split / f'{start:07d}.parquet'

    def save_frame(self, frame, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.parquet')
        pl.DataFrame(frame.to_dict(orient="list"), schema_overrides={c: pl.String for c in ("item_id", "query_id") if c in frame}).write_parquet(temporary)
        temporary.replace(path)

    def read_frame(self, path):
        frame = pd.DataFrame(pl.read_parquet(path).to_dict(as_series=False))
        for col in ('item_id', 'query_id'):
            if col in frame:
                frame[col] = frame[col].astype(object)
        if 'query_index' in frame:
            frame['query_index'] = frame['query_index'].astype('int64')
        return frame

    def prepare_cache(self):
        self.sources = {'bm25_title': ('item_title_raw', 'search_query'), 'bm25_description': ('item_description_raw', 'search_query'), 'bm25_infm': ('item_infm_params_text', 'search_infm_params_text')}
        self.names = list(self.sources) + self.dense_models
        self.order = [name for name in self.score_order if name in self.names]
        assert set(self.order) == set(self.names) and len(self.order) == len(self.names)
        stat = self.data_path.stat()
        code_hash = hashlib.sha256(b''.join((p.read_bytes() for p in sorted((self.root / 'src').rglob('*.py'))))).hexdigest()
        self.signature = dict(data=[str(self.data_path.resolve()), stat.st_size, stat.st_mtime_ns], queries={s: hashlib.sha256(pd.util.hash_pandas_object(q, index=True).to_numpy().tobytes()).hexdigest() for s, q in self.queries.items()}, code=code_hash, dense=self.dense_metadata, models=self.names, lemma=self.bm25_lemmatize, max_n=max(self.ns), batch=self.query_batch_size)
        self.cache = self.artifacts / hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / 'manifest.json').write_text(json.dumps(self.signature, indent=2))
        self.jobs = [(s, start) for s, q in self.queries.items() for start in range(0, len(q), self.query_batch_size)]
        self.geo = GeoFirstRetriever(n_candidates=max(self.ns)).fit(self.items)
        for split, start in self.jobs:
            path = self.cache_path('geo', split, start)
            if not path.exists():
                self.save_frame(self.geo.candidates(self.queries[split].iloc[start:start + self.query_batch_size]), path)
        print('Геопулы готовы:', self.cache)
        self.hybrid = HybridRetriever(distance_only=True).fit(self.items)

    def build(self, kinds=("retrieval", "scores")):
        if self.dense_models:
            from src.retrievers.dense import DenseRetriever
            import torch
        from time import perf_counter
        for name in self.names:
            pending = [(s, b) for s, b in self.jobs if any((not self.cache_path(kind, s, b, name).exists() for kind in kinds))]
            if not pending:
                print(name, 'в кеше')
                continue
            started = perf_counter()
            if name in self.sources:
                item_col, query_col = self.sources[name]
                corpus = self.read_item_text(item_col)
                model = BM25Retriever(item_text_column=item_col, query_text_column=query_col, lemmatize=self.bm25_lemmatize, batch_size=self.query_batch_size).fit(corpus)
            else:
                corpus = self.read_item_text('item_title_raw')
                model = DenseRetriever.load(Path(self.dense_index_dir) / name, device='cuda')
                expected = corpus.loc[corpus.item_title_raw.fillna('').str.strip().ne(''), 'item_id'].tolist()
                if model.item_ids_.tolist() != expected:
                    raise ValueError(f'{name}: сохранённый индекс не соответствует корпусу train.')
                model.batch_size = self.dense_batch_size
                del expected
            del corpus
            for i, (split, start) in enumerate(pending, 1):
                batch = self.queries[split].iloc[start:start + self.query_batch_size]
                path = self.cache_path('retrieval', split, start, name)
                if 'retrieval' in kinds and not path.exists():
                    self.save_frame(model.retrieve(batch, k=max(self.ns)), path)
                path = self.cache_path('scores', split, start, name)
                if 'scores' in kinds and not path.exists():
                    pool = self.read_frame(self.cache_path('geo', split, start))
                    result = pool[['query_index', 'item_id']].copy()
                    result['model_score'] = model.score_candidates(batch, pool)
                    self.save_frame(result, path)
                if i % 10 == 0 or i == len(pending):
                    print(name, f'{i}/{len(pending)} батчей, {perf_counter() - started:.0f} с', flush=True)
            del model
            gc.collect()
            if self.dense_models:
                torch.cuda.empty_cache()

    def late_rerank(self, base, feature, weight):
        if feature == 'none' or weight == 0 or base.empty:
            return base
        head = base.loc[base['rank'] <= self.quality_depth].copy()
        tail = base.loc[base['rank'] > self.quality_depth].copy()
        depth = head.groupby('query_index')['rank'].transform('max')
        signal = self.quality[feature].reindex(head.item_id).to_numpy()
        head['utility'] = 1 - (head['rank'] - 1) / np.maximum(depth - 1, 1) + weight * signal
        head = head.sort_values(['query_index', 'utility', 'rank'], ascending=[True, False, True], kind='stable')
        head['rank'] = head.groupby('query_index').cumcount() + 1
        return pd.concat([head, tail], ignore_index=True).sort_values(['query_index', 'rank'])

    def measure(self, result, target_sets):
        output = np.zeros((len(target_sets), len(self.ks)))
        for qi, g in result.groupby('query_index', sort=False):
            qi = int(qi)
            for j, k in enumerate(self.ks):
                output[qi, j] = len(set(g.loc[g['rank'] <= k, 'item_id']) & target_sets[qi]) / len(target_sets[qi])
        return output

    def evaluate(self, split, configs):
        arrays = {key: np.zeros((len(self.queries[split]), len(self.ks))) for key in configs}
        pool_stats = {key: np.zeros((len(self.queries[split]), 2)) for key in configs}
        for start in range(0, len(self.queries[split]), self.query_batch_size):
            q = self.queries[split].iloc[start:start + self.query_batch_size]
            truth = self.targets[split][start:start + len(q)]
            retrieved = {name: self.read_frame(self.cache_path('retrieval', split, start, name)) for name in self.names}
            scored = self.read_frame(self.cache_path('geo', split, start))
            for name in self.names:
                scores = self.read_frame(self.cache_path('scores', split, start, name)).rename(columns={'model_score': name})
                scored = scored.merge(scores, on=['query_index', 'item_id'], how='left', validate='one_to_one')
            bases = {}
            for approach, n, feature, weight in configs:
                pair = (approach, n)
                if pair not in bases:
                    if approach == 'text_then_geo':
                        self.hybrid.set_params(n_candidates=n)
                        pool = self.hybrid.combine(q, retrieved)
                        base = self.hybrid.rerank(pool, k=max(self.quality_depth, max(self.ks)))
                    else:
                        pool = scored.loc[scored.geo_rank <= n]
                        ranker = GeoFirstRetriever(n_candidates=n, fusion='tuple', score_order=tuple(self.order))
                        base = ranker.rerank(pool, k=max(self.quality_depth, max(self.ks)))
                    stats = np.zeros((len(q), 2))
                    for qi, g in pool.groupby('query_index', sort=False):
                        qi = int(qi)
                        ids = set(g.item_id)
                        stats[qi] = (len(ids & truth[qi]) / len(truth[qi]), len(ids))
                    bases[pair] = (base, stats)
                base, stats = bases[pair]
                key = (approach, n, feature, weight)
                arrays[key][start:start + len(q)] = self.measure(self.late_rerank(base, feature, weight), truth)
                pool_stats[key][start:start + len(q)] = stats
        rows = []
        for key, a in arrays.items():
            stats = pool_stats[key]
            row = dict(zip(['approach', 'N', 'feature', 'weight'], key))
            row.update({f'Recall@{k}': a[:, j].mean() for j, k in enumerate(self.ks)})
            row.update(pool_recall=stats[:, 0].mean(), mean_pool_size=stats[:, 1].mean(), nonempty_fraction=(stats[:, 1] > 0).mean())
            rows.append(row)
        return (pd.DataFrame(rows), arrays)

    def validate(self):
        approaches = ['text_then_geo', 'geo_then_tuple']
        baseline_configs = [(a, n, 'none', 0.0) for a in approaches for n in self.ns]
        self.val_base, _ = self.evaluate('val', baseline_configs)
        self.best_n = {a: int(self.val_base.loc[self.val_base.approach == a].sort_values([f'Recall@{self.select_k}', 'N'], ascending=[False, True]).iloc[0].N) for a in approaches}
        feature_configs = [(a, self.best_n[a], f, w) for a in approaches for f in ['rating', 'reviews', 'phone', 'messages', 'all'] for w in self.quality_weights]
        self.val_features, _ = self.evaluate('val', feature_configs)
        self.chosen = self.val_features.sort_values([f'Recall@{self.select_k}', 'weight'], ascending=[False, True]).drop_duplicates(['approach', 'feature'])
        self.selected = [(a, self.best_n[a], 'none', 0.0) for a in approaches]
        self.selected += [(r.approach, int(r.N), r.feature, float(r.weight)) for r in self.chosen.itertuples()]
        self.val_base.to_csv(self.cache / 'val_baselines.csv', index=False)
        self.val_features.to_csv(self.cache / 'val_features.csv', index=False)
        (self.cache / 'selected.json').write_text(json.dumps(self.selected, indent=2))
        display(self.val_base)
        display(self.chosen)
        return (self.val_base, self.chosen)

    def interval(self, delta):
        totals = np.bincount(self.group_ids, weights=delta)
        rng = np.random.default_rng(123)
        estimates = []
        for _ in range(2000):
            sample = rng.integers(0, len(totals), len(totals))
            estimates.append(totals[sample].sum() / self.group_count[sample].sum())
        return np.quantile(estimates, [0.025, 0.975])

    def test(self):
        self.comparison, self.per_query = self.evaluate('test', self.selected)
        self.group_ids = pd.factorize(self.queries['test'].search_query.fillna(''))[0]
        self.group_count = np.bincount(self.group_ids)
        j = self.ks.index(self.select_k)
        for i, key in enumerate(self.selected):
            a, n, f, w = key
            baseline = self.per_query[a, n, 'none', 0.0][:, j]
            delta = self.per_query[key][:, j] - baseline
            low, high = self.interval(delta)
            self.comparison.loc[i, 'delta'] = delta.mean()
            self.comparison.loc[i, 'delta_ci_low'] = low
            self.comparison.loc[i, 'delta_ci_high'] = high
        self.comparison.to_csv(self.cache / 'test_comparison.csv', index=False)
        raw = self.queries['test'].copy()
        for key, values in self.per_query.items():
            for col, k in enumerate(self.ks):
                raw[f"{'|'.join(map(str, key))}|Recall@{k}"] = values[:, col]
        self.save_frame(raw, self.cache / 'test_per_query.parquet')
        (self.cache / 'evaluation_settings.json').write_text(json.dumps(dict(NS=self.ns, KS=self.ks, select_k=self.select_k, score_order=self.order, quality_depth=self.quality_depth, quality_weights=self.quality_weights, rating_mean=self.rating_mean, review_scale=self.review_scale), indent=2))
        display(self.comparison)
        print('Результаты и кеш:', self.cache)
        return self.comparison

    def benchmark(self, *, items_path, queries_path, index_dir, build_missing_indexes,
                  approach, n, feature, weight, output_path):
        """Run fixed configuration on benchmark corpus; save validated answer.csv.

        Uses train location centers and feature statistics from prepare_data().
        Benchmark indices are separate from train indices. Models load one at a time.
        """
        import copy
        if approach not in {'text_then_geo', 'geo_then_tuple'}:
            raise ValueError('Unknown approach')
        if n < 50 or feature not in {'none', 'rating', 'reviews', 'phone', 'messages', 'all'}:
            raise ValueError('Invalid candidate count or feature')
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Weight must be finite and nonnegative')
        run = copy.copy(self)
        run.data_path = Path(items_path)
        run.dense_index_dir = Path(index_dir)
        if self.dense_models and run.dense_index_dir.resolve() == Path(self.dense_index_dir).resolve():
            raise ValueError('Use a separate benchmark index directory, not train indices')
        frame = pl.read_parquet(run.data_path, columns=['item_id', *self.geo_cols, *self.quality_cols])
        frame = frame.with_columns(pl.col('item_latitude', 'item_longitude').cast(pl.Float64))
        run.items = pd.DataFrame(frame.sort('item_id').to_dict(as_series=False))
        if run.items.item_id.isna().any() or run.items.item_id.duplicated().any():
            raise ValueError('Benchmark item_id must be unique and non-null')
        if not run.items.item_id.map(lambda x: isinstance(x, str) and bool(re.fullmatch('[0-9a-f]{16}', x))).all():
            raise ValueError('Invalid benchmark item_id')
        q = pd.DataFrame(pl.read_parquet(queries_path).to_dict(as_series=False))
        if (q.query_id.isna().any() or q.query_id.duplicated().any()
                or not q.query_id.map(lambda x: isinstance(x, str) and len(x) == 16).all()):
            raise ValueError('Benchmark query_id must be unique strings of length 16')
        run.queries = {'benchmark': q.join(self.centers, on='search_location_id')}
        run.ns = [int(n)]
        run.artifacts = self.artifacts / 'benchmark'
        run.prepare_quality()
        run.dense_metadata = {}
        for name in self.dense_models:
            from src.retrievers.dense import DenseRetriever
            import torch
            params = json.loads((Path(self.dense_index_dir)/name/'metadata.json').read_text())['params']
            params.update(device='cuda', batch_size=self.dense_batch_size)
            path = run.dense_index_dir / name
            if not all((path / f).is_file() for f in ('index.faiss', 'metadata.json')):
                if not build_missing_indexes:
                    raise FileNotFoundError(f'{path}: enable BUILD_BENCHMARK_INDEXES or attach benchmark indices')
                model = DenseRetriever(**params).fit(run.read_item_text('item_title_raw'))
                model.save(path)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            raw = (path/'metadata.json').read_bytes()
            saved = json.loads(raw)['params']
            for key in ('model', 'revision', 'max_length', 'item_text_column', 'query_text_column', 'query_instruction', 'dtype'):
                if saved.get(key) != params.get(key):
                    raise ValueError(f'{name}: benchmark index differs in {key}; use a new index directory')
            stat = (path/'index.faiss').stat()
            run.dense_metadata[name] = [hashlib.sha256(raw).hexdigest(), stat.st_size, stat.st_mtime_ns]
        run.prepare_cache()
        run.build(kinds=('retrieval',) if approach == 'text_then_geo' else ('scores',))
        predictions = [[] for _ in range(len(q))]
        for start in range(0, len(q), self.query_batch_size):
            batch = run.queries['benchmark'].iloc[start:start+self.query_batch_size]
            if approach == 'text_then_geo':
                results = {name: run.read_frame(run.cache_path('retrieval','benchmark',start,name)) for name in run.names}
                run.hybrid.set_params(n_candidates=n)
                base = run.hybrid.rerank(run.hybrid.combine(batch,results), k=max(self.quality_depth,50))
            else:
                pool = run.read_frame(run.cache_path('geo','benchmark',start))
                for name in run.names:
                    scores = run.read_frame(run.cache_path('scores','benchmark',start,name)).rename(columns={'model_score':name})
                    pool = pool.merge(scores,on=['query_index','item_id'],how='left',validate='one_to_one')
                ranker = GeoFirstRetriever(n_candidates=n,fusion='tuple',score_order=tuple(run.order))
                base = ranker.rerank(pool,k=max(self.quality_depth,50))
            result = run.late_rerank(base,feature,weight)
            for qi,group in result.loc[result['rank']<=50].groupby('query_index',sort=False):
                predictions[start+int(qi)] = group.sort_values('rank').item_id.tolist()
        return write_answer(q.query_id.tolist(), predictions, run.items.item_id.tolist(), output_path)


def write_answer(query_ids, predictions, item_ids, output_path):
    """Validate identifiers without casting or changing case; write UTF-8 CSV."""
    import re
    if len(query_ids) != len(predictions) or len(set(query_ids)) != len(query_ids):
        raise ValueError('Exactly one prediction list per unique query_id is required')
    if not all(isinstance(q, str) and len(q) == 16 for q in query_ids):
        raise ValueError('query_id must be a string of length 16')
    allowed = set(item_ids)
    for ids in predictions:
        if len(ids) > 50 or len(ids) != len(set(ids)):
            raise ValueError('At most 50 distinct item_id values per query')
        if any(not isinstance(i, str) or not re.fullmatch('[0-9a-f]{16}', i) or i not in allowed for i in ids):
            raise ValueError('Prediction contains an invalid or unknown benchmark item_id')
    answer = pd.DataFrame({'query_id':query_ids, 'answer':[' '.join(ids) for ids in predictions]})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.csv')
    answer.to_csv(temporary,index=False,encoding='utf-8',sep=',')
    restored = pd.read_csv(temporary,dtype=str,keep_default_na=False)
    if list(restored.columns) != ['query_id','answer'] or not restored.equals(answer):
        raise ValueError('CSV round-trip validation failed')
    temporary.replace(path)
    print(f'Saved {len(answer)} queries: {path}')
    return answer
