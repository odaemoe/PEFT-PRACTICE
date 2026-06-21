"""
train_eval.py — унифицированный цикл обучения и оценки одной модели.

Для одного метода на одном датасете:
  • обучает модель (HuggingFace Trainer),
  • замеряет точность (accuracy) и F1 на тесте,
  • фиксирует время обучения и пик потребления VRAM,
  • сохраняет историю логов (для построения кривых обучения),
  • при необходимости сохраняет адаптер на диск (для последующей загрузки).
"""

import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

import config
from models import create_model, count_trainable


def _compute_metrics(eval_pred):
    """Метрики для Trainer: accuracy и F1 (бинарная)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def train_and_eval(method_name, method_spec, data, tokenizer, save_adapter=False):
    """Обучает и оценивает один метод на одном датасете.

    Возвращает словарь с метриками и историей логов.
    """
    model = create_model(method_spec)
    trainable, total = count_trainable(model)
    print(f"   Обучаемые параметры: {trainable:,} ({100 * trainable / total:.2f}%)")

    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    out_dir = os.path.join(config.RESULTS_DIR, "checkpoints", safe_name)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=method_spec["lr"],          # свой LR на каждый метод
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="epoch",                    # оценка после каждой эпохи -> кривые
        save_strategy="no",                       # промежуточные чекпойнты не сохраняем
        logging_steps=config.LOGGING_STEPS,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=config.SEED,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_metrics=_compute_metrics,
        data_collator=collator,
    )

    # Замер пика VRAM и времени обучения.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024 ** 3
        if torch.cuda.is_available()
        else 0.0
    )

    final = trainer.evaluate()

    # Сохранение адаптера/модели (требование п. 1.4 задания).
    if save_adapter:
        adapter_dir = os.path.join(config.ADAPTERS_DIR, safe_name)
        os.makedirs(adapter_dir, exist_ok=True)
        trainer.save_model(adapter_dir)
        print(f"   Адаптер сохранён в {adapter_dir}")

    return {
        "method": method_name,
        "accuracy": final["eval_accuracy"],
        "f1": final["eval_f1"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "train_time_sec": train_time,
        "peak_vram_gb": peak_vram,
        "log_history": trainer.state.log_history,  # для кривых обучения
    }
