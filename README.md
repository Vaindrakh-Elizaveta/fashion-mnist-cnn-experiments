# Fashion-MNIST CNN experiments

Проект посвящён исследованию классификации Fashion-MNIST с помощью небольшой CNN и сравнению моделей, обученных на полной и сокращённой обучающих выборках.

## Текущий статус

В проекте реализованы:

- базовая модель на полной обучающей выборке `F`;
- базовая модель на сокращённой выборке `F-A`;
- стратифицированное разбиение `F-A` и `F-B`;
- двухэтапное обучение с early stopping;
- автоматический выбор GPU CUDA;
- сохранение checkpoint, истории обучения, метрик и матриц ошибок;
- проверка устойчивости ошибок при пяти запусках CNN с разными seed.
- EDNNA для итеративного отбора дополнительных изображений;
- равномерный случайный отбор такого же количества изображений;
- случайный отбор по долям ошибок истинных классов;
- итоговое сравнение Accuracy, Macro-F1 и направленных ошибок.

Для EDNNA реализованы критерии остановки: достижение целевой validation accuracy, исчерпание бюджета дополнительных изображений и отсутствие подходящих кандидатов в `F-B`. В текущих экспериментах временный целевой порог accuracy задавался как `0.92`, а максимальный бюджет составлял 30% от исходной выборки `F-A`.

## Данные и разбиение

Fashion-MNIST разделён следующим образом:

- исходная train-выборка — 60 000 изображений;
- train — 54 000 изображений;
- validation — 6 000 изображений;
- test — 10 000 изображений.

Для эксперимента с ограниченным объёмом данных исходная train-выборка разделена стратифицированно:

- `F-A` — 10 800 изображений, 20% train;
- `F-B` — 43 200 изображений, 80% train.

Разбиение выполнено с фиксированным `seed = 42`. Validation и test не входят в `F-A` и `F-B`.

Нормализация рассчитывается отдельно для соответствующей обучающей выборки и не использует test-данные.

## Архитектура и обучение

Используется CNN из файла [`src/model.py`](src/model.py):

- два сверточных блока `Conv2d + BatchNorm + ReLU + MaxPool`;
- полносвязный слой на 64 признака;
- `Dropout(0.3)`;
- выходной слой на 10 классов.

Параметры обучения:

- оптимизатор — Adam;
- начальный learning rate — `0.001`;
- learning rate второго этапа — `0.0001`;
- weight decay — `0.0001`;
- batch size — `128`;
- максимум — 100 эпох на каждый этап;
- early stopping — после 10 эпох без улучшения Validation Loss;
- критерий выбора checkpoint — минимальный Validation Loss.

После первого этапа learning rate уменьшается в 10 раз, затем обучение продолжается с тем же критерием остановки.

## Результаты базовых моделей

| Эксперимент | Train size | Validation accuracy | Test accuracy | Macro-F1 | Лучшая эпоха |
|---|---:|---:|---:|---:|---:|
| Полная выборка `F` | 54 000 | 93.12% | 92.00% | 91.97% | 24 |
| Сокращённая выборка `F-A` | 10 800 | 90.27% | 89.30% | 89.27% | 26 |

Для обеих моделей наиболее сложным классом является `Shirt`. На полной модели суммарное число ошибок между `T-shirt/top` и `Shirt` составляет 188, для `F-A` — 223.

## EDNNA и случайные стратегии

Дополнительные изображения выбирались из `F-B`, а затем модель заново обучалась на расширенной выборке `F-A`. Для сравнения использовалось одинаковое количество добавленных изображений — 3240, то есть 30% от исходного размера `F-A`.

| Эксперимент | Размер train | Test Accuracy | Macro-F1 |
|---|---:|---:|---:|
| `F-A` baseline | 10 800 | 0.8930 | 0.892675 |
| EDNNA +1000 | 11 800 | 0.8927 | 0.893094 |
| EDNNA +20% | 12 960 | 0.8996 | 0.899315 |
| EDNNA +30% | 14 040 | 0.9000 | 0.899794 |
| Random uniform | 14 040 | 0.8973 | 0.897004 |
| Random by error class | 14 040 | 0.8988 | 0.899388 |

EDNNA с увеличением выборки на 30% показал лучший результат среди методов расширения `F-A`. По сравнению с базовой моделью Accuracy выросла с 0.8930 до 0.9000, а Macro-F1 — с 0.892675 до 0.899794.

Для направленных ошибок между `Shirt` и `T-shirt/top` получены следующие результаты:

| Эксперимент | `Shirt → T-shirt/top` | `T-shirt/top → Shirt` | Всего ошибок |
|---|---:|---:|---:|
| `F-A` baseline | 120 | 103 | 223 |
| EDNNA +1000 | 103 | 121 | 224 |
| EDNNA +20% | 114 | 103 | 217 |
| EDNNA +30% | 90 | 105 | 195 |
| Random uniform | 104 | 104 | 208 |
| Random by error class | 90 | 110 | 200 |

EDNNA +30% уменьшил суммарное количество ошибок между двумя наиболее сложными классами с 223 до 195. Это показывает, что направленный отбор EDNNA даёт преимущество по сравнению со случайным добавлением такого же количества изображений. При этом разница между EDNNA и случайным отбором по долям ошибок невелика, поэтому для строгого вывода желательно повторить эксперименты с несколькими значениями seed.

## Проверка устойчивости ошибок

Одна архитектура CNN была обучена пять раз с seed `42, 43, 44, 45, 46`. Все модели проверялись на одной контрольной validation-выборке из 6 000 изображений.

Для каждого изображения вычислялась частота ошибки:

```text
q(x) = число моделей, ошибшихся на x / 5
```

Результаты:

- Accuracy отдельных запусков: от 90.15% до 90.57%;
- множество изображений, на которых ошиблась хотя бы одна модель: 947 из 6 000;
- среднее `q(x)` по этому множеству: `0.6097`;
- значения `q_average` по классам: от `0.4545` до `0.7294`.

Таким образом, общая точность моделей достаточно стабильна, но полная модельная независимость ошибок не подтверждена: ошибки отдельных изображений заметно зависят от случайной инициализации CNN. Поэтому результаты EDNNA требуют дополнительного согласования критерия отбора устойчивых ошибок.

## Структура проекта

```text
configs/       конфигурации экспериментов
data/          локальный Fashion-MNIST; не хранится в Git
experiments/   checkpoint, история, метрики, графики и индексы
notebooks/     исследовательский Jupyter Notebook
src/           загрузка данных, модель, обучение и предсказания
```

Основные файлы:

- [`notebooks/01_fashion_mnist_exploration.ipynb`](notebooks/01_fashion_mnist_exploration.ipynb) — основной эксперимент;
- [`src/model.py`](src/model.py) — архитектура CNN;
- [`src/train.py`](src/train.py) — обучение и двухэтапный scheduler;
- [`src/data.py`](src/data.py) — загрузка и подготовка данных;
- [`configs/fashion_mnist_baseline.json`](configs/fashion_mnist_baseline.json) — параметры запуска.

## Установка

Из корневой папки репозитория:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Для запуска на NVIDIA GPU установите CUDA-сборки PyTorch:

```powershell
python -m pip install --force-reinstall `
  "torch==2.6.0+cu124" `
  "torchvision==0.21.0+cu124" `
  --index-url https://download.pytorch.org/whl/cu124 `
  --extra-index-url https://pypi.org/simple
```

Проверка GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Ожидается `True` и название NVIDIA GPU.

## Запуск ноутбука

Откройте:

```text
notebooks/01_fashion_mnist_exploration.ipynb
```

В VS Code выберите kernel из окружения `.venv`, затем выполните ячейки сверху вниз. Устройство выбирается автоматически:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Основные результаты сохраняются в:

```text
experiments/fashion_mnist/baseline/
experiments/fashion_mnist/fa_baseline/
experiments/fashion_mnist/model_stability/
experiments/fashion_mnist/ednna/
experiments/fashion_mnist/ednna_plus20/
experiments/fashion_mnist/ednna_plus30/
experiments/fashion_mnist/random_selection/random_uniform/
experiments/fashion_mnist/random_selection/random_by_error_class/
experiments/fashion_mnist/comparison/
```

В частности, проверка устойчивости ошибок сохраняется в:

```text
experiments/fashion_mnist/model_stability/metrics/q_average_by_class.csv
experiments/fashion_mnist/model_stability/metrics/per_image_error_frequency.csv
experiments/fashion_mnist/model_stability/figures/q_average_by_class.png
```

Сводные таблицы сравнения сохраняются в:

```text
experiments/fashion_mnist/comparison/all_models_comparison_with_random.csv
experiments/fashion_mnist/comparison/directed_errors_comparison_with_random.csv
```
