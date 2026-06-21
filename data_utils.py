"""
data_utils.py — загрузка, унификация и токенизация датасетов.

Все датасеты приводятся к единому виду: две колонки — "text" (текст отзыва)
и "label" (0 = негатив, 1 = позитив). Это позволяет один и тот же код
обучения применять к любому датасету из config.DATASETS.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

import config


def get_tokenizer():
    """Возвращает токенизатор базовой модели."""
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)


def _prepare_split(ds, spec, n_samples):
    """Готовит один сплит: фильтрация -> подвыборка -> унификация колонок."""
    # Фильтрация по меткам (нужно для датасетов с числом классов > 2).
    if spec["label_filter"] is not None:
        ds = ds.filter(lambda x: x[spec["label_col"]] in spec["label_filter"])

    # Перемешивание с фиксированным зерном и взятие сбалансированной подвыборки.
    ds = ds.shuffle(seed=config.SEED).select(range(min(n_samples, len(ds))))

    # Приведение колонок к именам "text" / "label".
    if spec["text_col"] != "text":
        ds = ds.rename_column(spec["text_col"], "text")
    if spec["label_col"] != "label":
        ds = ds.rename_column(spec["label_col"], "label")

    # Переотображение меток в {0, 1}, если нужно.
    if spec["label_map"] is not None:
        label_map = spec["label_map"]
        ds = ds.map(lambda x: {"label": label_map[x["label"]]})

    # Оставляем только нужные колонки.
    keep = {"text", "label"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def load_one_dataset(name, quick=False):
    """Загружает и готовит один датасет по имени из config.DATASETS."""
    spec = config.DATASETS[name]
    if spec["config"]:
        raw = load_dataset(spec["hf_path"], spec["config"])
    else:
        raw = load_dataset(spec["hf_path"])

    n_train = config.QUICK_TRAIN_SIZE if quick else config.TRAIN_SIZE
    n_test = config.QUICK_TEST_SIZE if quick else config.TEST_SIZE

    train = _prepare_split(raw[spec["train_split"]], spec, n_train)
    test = _prepare_split(raw[spec["test_split"]], spec, n_test)
    return {"train": train, "test": test}


def tokenize_dataset(ds, tokenizer):
    """Токенизирует датасет (колонка text -> input_ids / attention_mask)."""
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.MAX_LENGTH)

    return ds.map(_tok, batched=True, remove_columns=["text"])


def load_all(names, tokenizer, quick=False):
    """Загружает и токенизирует все указанные датасеты.

    Возвращает словарь вида:
        {"IMDb": {"train": <tokenized>, "test": <tokenized>}, "SST-2": {...}}
    """
    out = {}
    for name in names:
        print(f"  • Загрузка датасета {name}...")
        raw = load_one_dataset(name, quick=quick)
        out[name] = {
            "train": tokenize_dataset(raw["train"], tokenizer),
            "test": tokenize_dataset(raw["test"], tokenizer),
        }
        print(f"    {name}: {len(out[name]['train'])} train / {len(out[name]['test'])} test")
    return out
