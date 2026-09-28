# Поиск объявлений по текстовому запросу

В проекте есть BM25, TF-IDF и dense-ретривер (BGE-M3 и Giga-480M), а также ноутбуки для локальной оценки Recall@K. Реализации возвращают `query_index`, `item_id`, `rank` и `score`.

Для запуска локальных экспериментов положите `train.parquet` в `data/`. Исходные parquet-файлы не входят в Git. Установите зависимости командой `uv sync`; для dense-моделей — `uv sync --extra dense`.

Kaggle-ноутбук [Dense_pipeline_kaggle.ipynb](notebooks/Dense_pipeline_kaggle.ipynb) содержит настройки клонирования репозитория и пути к датасету. Для него нужны GPU и доступ к интернету при первой загрузке моделей.
