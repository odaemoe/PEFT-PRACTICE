"""
config.py — централизованное управление гиперпараметрами эксперимента.

Здесь задаются: базовая модель, список датасетов, список сравниваемых
методов (Full Fine-Tuning, LoRA разных рангов, IA3, Prompt Tuning) и
гиперпараметры обучения. Меняя значения в этом файле, можно управлять всем
экспериментом, не трогая остальной код.
"""

# --- Базовая модель ---
MODEL_NAME = "roberta-base"   # backbone-трансформер (~125 млн параметров)
NUM_LABELS = 2                # бинарная классификация тональности (pos / neg)
MAX_LENGTH = 256              # макс. длина последовательности при токенизации
SEED = 42                     # фиксированное случайное зерно для воспроизводимости

# --- Объём данных ---
# Подвыборки нужны, чтобы эксперименты укладывались в разумное время на Colab.
# Для финального прогона значения можно увеличить (например, 5000 / 1000).
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
# Каждый датасет описывается тем, как его грузить из HuggingFace и как привести
# к единому виду (колонки text / label).
#   train_split / test_split — какие сплиты использовать (у SST-2 тестовые метки
#   скрыты, поэтому в качестве тестовой берём validation).
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
        # namespaced-репозиторий (parquet) — грузится одинаково на datasets 2.x и 5.x,
        # в отличие от устаревшего канонического "glue".
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
# type — тип метода: full / lora / ia3 / prompt / prefix
# lr   — learning rate. ВАЖНО: оптимальный LR у разных методов отличается на
#        порядки. Full FT и LoRA любят малый LR, а IA3 и Prompt Tuning требуют
#        существенно большего, иначе их немногочисленные параметры почти не
#        сдвигаются. Именно из-за единого малого LR (2e-5) в курсовой работе
#        методы IA3 и Prompt Tuning «не завелись» — здесь это исправлено.
METHODS = {
    "Full FT":       {"type": "full",   "lr": 2e-5},
    "LoRA (r=4)":    {"type": "lora",   "lr": 2e-4, "lora_rank": 4},
    "LoRA (r=8)":    {"type": "lora",   "lr": 2e-4, "lora_rank": 8},
    "LoRA (r=16)":   {"type": "lora",   "lr": 2e-4, "lora_rank": 16},
    "IA3":           {"type": "ia3",    "lr": 3e-3},
    "Prompt Tuning": {"type": "prompt", "lr": 1e-2, "num_virtual_tokens": 20},

    # Альтернатива третьему PEFT-методу: если Prompt Tuning на вашей версии
    # библиотеки PEFT падает с ошибкой при классификации — закомментируйте
    # строку "Prompt Tuning" выше и раскомментируйте строку ниже.
    # "Prefix Tuning": {"type": "prefix", "lr": 1e-3, "num_virtual_tokens": 20},
}

# --- Пути для сохранения результатов ---
RESULTS_DIR = "results"     # таблицы (.csv/.md) и графики (.png)
ADAPTERS_DIR = "adapters"   # сохранённые PEFT-адаптеры (для загрузки/деплоя)
