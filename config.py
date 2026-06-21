"""
config.py — централизованное управление гиперпараметрами эксперимента.

"""

# --- Базовая модель ---
MODEL_NAME = "roberta-base"   # backbone-трансформер (~125 млн параметров)
NUM_LABELS = 2                # бинарная классификация тональности (pos / neg)
MAX_LENGTH = 256              # макс. длина последовательности при токенизации
SEED = 42                     # фиксированное случайное зерно для воспроизводимости

# --- Объём данных ---
TRAIN_SIZE = 2000
TEST_SIZE = 1000

# Быстрый smoke-test (для проверки, что весь pipeline работает за пару минут).
QUICK_TRAIN_SIZE = 400
QUICK_TEST_SIZE = 200

# --- Общие гиперпараметры обучения ---
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LOGGING_STEPS = 25            # как часто логировать training loss (для кривых обучения)

# --- Параметры LoRA ---
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# --- Датасеты ---
DATASETS = {
    "IMDb": {
        "hf_path": "stanfordnlp/imdb",
        "config": None,
        "train_split": "train",
        "test_split": "test",
        "text_col": "text",
        "label_col": "label",
        "label_filter": None,   # фильтр меток (если в датасете >2 классов)
        "label_map": None,      # переотображение меток в {0, 1}
    },
    "SST-2": {
        "hf_path": "stanfordnlp/sst2",
        "config": None,
        "train_split": "train",
        "test_split": "validation",   # у SST-2 метки теста скрыты (-1)
        "text_col": "sentence",
        "label_col": "label",
        "label_filter": None,
        "label_map": None,
    },
}

# --- Сравниваемые методы ---
METHODS = {
    "Full FT":       {"type": "full",   "lr": 2e-5},
    "LoRA (r=4)":    {"type": "lora",   "lr": 2e-4, "lora_rank": 4},
    "LoRA (r=8)":    {"type": "lora",   "lr": 2e-4, "lora_rank": 8},
    "LoRA (r=16)":   {"type": "lora",   "lr": 2e-4, "lora_rank": 16},
    "IA3":           {"type": "ia3",    "lr": 3e-3},
    "Prompt Tuning": {"type": "prompt", "lr": 1e-2, "num_virtual_tokens": 20},

}

# --- Пути для сохранения результатов ---
RESULTS_DIR = "results"     # таблицы (.csv/.md) и графики (.png)
ADAPTERS_DIR = "adapters"   # сохранённые PEFT-адаптеры (для загрузки/деплоя)
