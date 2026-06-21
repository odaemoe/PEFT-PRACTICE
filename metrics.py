"""
metrics.py — построение сводных таблиц и графиков по результатам экспериментов.

Создаёт:
  • сводную таблицу метрик (summary_<датасет>.csv и .md),
  • кривые обучения всех методов на одном полотне (curves_<датасет>.png),
  • диаграммы сравнения методов по качеству и ресурсам (compare_<датасет>.png).
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

import config


def results_to_dataframe(results):
    """Превращает список результатов в табличку pandas."""
    rows = []
    for r in results:
        rows.append({
            "Метод": r["method"],
            "Accuracy": round(r["accuracy"], 4),
            "F1": round(r["f1"], 4),
            "Обучаемые параметры": r["trainable_params"],
            "Доля параметров, %": round(r["trainable_pct"], 3),
            "Время обучения, с": round(r["train_time_sec"], 1),
            "Пик VRAM, ГБ": round(r["peak_vram_gb"], 2),
        })
    return pd.DataFrame(rows)


def save_summary_table(results, dataset_name, out_dir=None):
    """Сохраняет сводную таблицу в CSV и Markdown."""
    out_dir = out_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    df = results_to_dataframe(results)

    base = os.path.join(out_dir, f"summary_{dataset_name}")
    df.to_csv(base + ".csv", index=False, encoding="utf-8-sig")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(f"# Сводная таблица результатов — {dataset_name}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    return df


def plot_training_curves(results, dataset_name, out_dir=None):
    """Кривые обучения (training loss по шагам) всех методов на одном полотне."""
    out_dir = out_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(11, 6))
    plotted = False
    for r in results:
        steps, losses = [], []
        for log in r["log_history"]:
            if "loss" in log and "step" in log:
                steps.append(log["step"])
                losses.append(log["loss"])
        if steps:
            plt.plot(steps, losses, marker="o", markersize=3, linewidth=2, label=r["method"])
            plotted = True

    plt.xlabel("Шаг обучения", fontsize=13)
    plt.ylabel("Training loss", fontsize=13)
    plt.title(f"Кривые обучения методов — {dataset_name}", fontsize=15, fontweight="bold")
    if plotted:
        plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, f"curves_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_method_comparison(results, dataset_name, out_dir=None):
    """Диаграммы сравнения методов: качество (Accuracy/F1), доля параметров, ресурсы."""
    out_dir = out_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    names = [r["method"] for r in results]
    acc = [r["accuracy"] * 100 for r in results]
    f1 = [r["f1"] * 100 for r in results]
    pct = [r["trainable_pct"] for r in results]
    vram = [r["peak_vram_gb"] for r in results]
    times = [r["train_time_sec"] / 60 for r in results]

    x = range(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (1) Качество: Accuracy и F1
    w = 0.38
    axes[0].bar([i - w / 2 for i in x], acc, w, label="Accuracy, %", color="#1f77b4")
    axes[0].bar([i + w / 2 for i in x], f1, w, label="F1, %", color="#2ca02c")
    axes[0].set_title("Качество классификации", fontweight="bold")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 100)
    axes[0].legend()

    # (2) Доля обучаемых параметров (логарифмическая шкала: от <1% до 100%)
    axes[1].bar(list(x), pct, color="#ff7f0e")
    axes[1].set_title("Доля обучаемых параметров", fontweight="bold")
    axes[1].set_ylabel("% от всех параметров")
    axes[1].set_yscale("log")

    # (3) Ресурсы: время обучения и пик VRAM
    ax3 = axes[2]
    ax3.bar([i - w / 2 for i in x], times, w, label="Время, мин", color="#9467bd")
    ax3.set_ylabel("Время обучения, мин", color="#9467bd")
    ax3.set_title("Вычислительные ресурсы", fontweight="bold")
    ax3b = ax3.twinx()
    ax3b.bar([i + w / 2 for i in x], vram, w, label="Пик VRAM, ГБ", color="#d62728")
    ax3b.set_ylabel("Пик VRAM, ГБ", color="#d62728")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Сравнение методов PEFT — {dataset_name}", fontsize=16, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(out_dir, f"compare_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
