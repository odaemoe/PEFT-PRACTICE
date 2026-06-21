"""
models.py — инициализация базовой модели и подключение PEFT-конфигураций.

Функция create_model по описанию метода из config.METHODS возвращает готовую
к обучению модель: либо полную (Full Fine-Tuning), либо обёрнутую в один из
PEFT-методов (LoRA / IA3 / Prompt Tuning / Prefix Tuning).
"""

from transformers import AutoModelForSequenceClassification
from peft import (
    LoraConfig,
    IA3Config,
    PromptTuningConfig,
    PrefixTuningConfig,
    get_peft_model,
    TaskType,
)

import config


def count_trainable(model):
    """Возвращает (число обучаемых параметров, общее число параметров)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _build_peft_config(method_spec):
    """Собирает PEFT-конфигурацию под конкретный метод."""
    mtype = method_spec["type"]

    if mtype == "lora":
        # LoRA: низкоранговые адаптеры на проекциях query/value механизма внимания.
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=method_spec["lora_rank"],
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=["query", "value"],
        )

    if mtype == "ia3":
        # IA3: обучаемые векторы масштабирования на ключах, значениях и входе FFN.
        # Это «каноничная» конфигурация IA3 (l_k, l_v, l_ff) применительно к RoBERTa.
        return IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=["key", "value", "intermediate.dense"],
            feedforward_modules=["intermediate.dense"],
        )

    if mtype == "prompt":
        # Prompt Tuning: обучаемые виртуальные токены на входе, сама модель заморожена.
        return PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=method_spec.get("num_virtual_tokens", 20),
        )

    if mtype == "prefix":
        # Prefix Tuning: обучаемые префиксы к ключам/значениям внимания.
        return PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=method_spec.get("num_virtual_tokens", 20),
            prefix_projection=True,
        )

    raise ValueError(f"Неизвестный тип метода: {mtype}")


def create_model(method_spec):
    """Создаёт модель для заданного метода (config.METHODS[...])."""
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=config.NUM_LABELS
    )

    # Full Fine-Tuning — обучаются все параметры, PEFT не подключается.
    if method_spec["type"] == "full":
        return model

    # Иначе замораживаем базовую модель и добавляем PEFT-адаптер.
    peft_config = _build_peft_config(method_spec)
    model = get_peft_model(model, peft_config)
    return model
