"""
main.py — точка входа для запуска экспериментов.

Для каждого датасета обучает все методы из config.METHODS, собирает метрики,
строит сводные таблицы и графики. Каждый эксперимент обёрнут в try/except:
если какой-то метод падает, остальные всё равно отрабатывают, а сбой
фиксируется (это само по себе является результатом исследования).

Запуск из терминала:
    python main.py                 # полный прогон по всем датасетам и методам
    python main.py --quick         # быстрый smoke-test на малых данных
    python main.py --datasets SST-2          # только один датасет
    python main.py --save-adapters           # дополнительно сохранить адаптеры
"""

import argparse
import random

import numpy as np
import torch

import config
import metrics
from data_utils import get_tokenizer, load_all
from train_eval import train_and_eval


def set_seed(seed):
    """Фиксирует случайное зерно во всех библиотеках для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(datasets=None, methods=None, quick=False, save_adapters=False):
    """Запускает полный цикл экспериментов и возвращает собранные результаты."""
    set_seed(config.SEED)
    datasets = datasets or list(config.DATASETS.keys())
    methods = methods or config.METHODS

    print("Инициализация токенизатора и загрузка данных...")
    tokenizer = get_tokenizer()
    data_all = load_all(datasets, tokenizer, quick=quick)

    all_results = {}
    for ds_name in datasets:
        print(f"\n{'=' * 64}\nДАТАСЕТ: {ds_name}\n{'=' * 64}")
        ds_results = []

        for m_name, m_spec in methods.items():
            print(f"\n--- {m_name} на {ds_name} ---")
            try:
                res = train_and_eval(
                    m_name, m_spec, data_all[ds_name], tokenizer,
                    save_adapter=save_adapters,
                )
                ds_results.append(res)
                print(
                    f"   Accuracy={res['accuracy']:.3f}  F1={res['f1']:.3f}  "
                    f"params={res['trainable_pct']:.2f}%  "
                    f"time={res['train_time_sec'] / 60:.1f}мин  "
                    f"VRAM={res['peak_vram_gb']:.2f}ГБ"
                )
            except Exception as e:  # noqa: BLE001 — намеренно ловим любой сбой метода
                print(f"   ОШИБКА метода '{m_name}': {e}")

        all_results[ds_name] = ds_results

        if ds_results:
            df = metrics.save_summary_table(ds_results, ds_name)
            metrics.plot_training_curves(ds_results, ds_name)
            metrics.plot_method_comparison(ds_results, ds_name)
            print(f"\nСводная таблица по {ds_name}:")
            print(df.to_string(index=False))
            print(f"\nГрафики и таблицы сохранены в папку '{config.RESULTS_DIR}/'.")

    print("\nГотово. Все эксперименты завершены.")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение PEFT-методов для анализа тональности.")
    parser.add_argument("--quick", action="store_true", help="быстрый smoke-test на малых данных")
    parser.add_argument("--datasets", nargs="*", default=None, help="список датасетов (по умолчанию все)")
    parser.add_argument("--save-adapters", action="store_true", help="сохранять обученные адаптеры")
    args = parser.parse_args()

    run(datasets=args.datasets, quick=args.quick, save_adapters=args.save_adapters)
