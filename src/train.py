import json
from pathlib import Path

import torch


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """
    Одна эпоха обучения модели.
    """

    # Режим обучения:
    # активируется Dropout и BatchNorm обучается
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in train_loader:
        # Переносим данные на CPU или GPU
        images = images.to(device)
        labels = labels.to(device)

        # Обнуляем градиенты с предыдущего шага
        optimizer.zero_grad()

        # Получаем предсказания модели
        logits = model(images)

        # Вычисляем ошибку
        loss = criterion(logits, labels)

        # Вычисляем градиенты
        loss.backward()

        # Обновляем параметры модели
        optimizer.step()

        # Размер текущей группы изображений
        batch_size = images.size(0)

        # Накапливаем ошибку
        total_loss += loss.item() * batch_size

        # Получаем предсказанный класс
        predictions = logits.argmax(dim=1)

        # Считаем правильные ответы
        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    """
    Оценка модели на validation или test.
    """

    # Режим оценки:
    # Dropout отключается, BatchNorm использует
    # уже рассчитанные статистики
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # Во время оценки градиенты не нужны
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def fit(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path=None,
    history_path=None,
    checkpoint_metadata=None,
):
    """
    Полный цикл обучения модели.

    Лучшая модель выбирается по validation accuracy.
    """

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_accuracy = float("-inf")
    best_epoch = None

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    for epoch in range(1, epochs + 1):

        # Обучение на train
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Проверка на validation
        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        # Сохраняем результаты текущей эпохи
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(
            validation_accuracy
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        # Проверяем, стала ли модель лучше
        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy
            best_epoch = epoch

            if checkpoint_path is not None:

                checkpoint = {
                    "epoch": epoch,

                    # Веса модели
                    "model_state_dict": (
                        model.state_dict()
                    ),

                    # Состояние оптимизатора Adam
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),

                    # Метрики лучшей модели
                    "validation_loss": validation_loss,
                    "validation_accuracy": (
                        validation_accuracy
                    ),

                    # История на момент сохранения
                    "history": history,
                }

                # Добавляем seed, нормализацию,
                # архитектуру и параметры обучения
                if checkpoint_metadata is not None:
                    checkpoint.update(
                        checkpoint_metadata
                    )

                torch.save(
                    checkpoint,
                    checkpoint_path,
                )

    # Сохраняем полную историю после всех эпох
    if history_path is not None:
        history_path = Path(history_path)

        history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        history_path.write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print("Лучшая эпоха:", best_epoch)
    print(
        "Лучшая validation accuracy:",
        best_validation_accuracy,
    )

    return history


import copy
import json
from pathlib import Path

import torch


def fit_two_stage(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    epochs=100,
    patience=10,
    min_delta=1e-4,
    lr_factor=0.1,
    checkpoint_path=None,
    history_path=None,
    checkpoint_metadata=None,
):
    """
    Двухэтапное обучение модели.

    Этап 1:
        обучение с исходным learning rate.

    Этап 2:
        загрузка лучшей модели с этапа 1,
        уменьшение learning rate в 10 раз,
        продолжение обучения.

    На каждом этапе используется early stopping
    по Validation Loss.
    """

    history = {
        "epoch": [],
        "stage": [],
        "learning_rate": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    best_validation_loss = float("inf")
    best_validation_accuracy = None
    best_epoch = None
    best_stage = None

    best_model_state = None
    best_optimizer_state = None

    for stage in [1, 2]:

        # Перед вторым этапом возвращаемся
        # к лучшей модели первого этапа
        if stage == 2:
            if best_model_state is None:
                raise RuntimeError(
                    "Не найдена лучшая модель первого этапа."
                )

            model.load_state_dict(
                best_model_state
            )

            if best_optimizer_state is not None:
                optimizer.load_state_dict(
                    best_optimizer_state
                )

            # Уменьшаем learning rate
            # в 10 раз
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_factor

            print()
            print(
                "Этап 2: learning rate уменьшен до",
                optimizer.param_groups[0]["lr"],
            )

        bad_epochs = 0

        for _ in range(epochs):

            global_epoch = len(
                history["epoch"]
            ) + 1

            train_loss, train_accuracy = (
                train_one_epoch(
                    model=model,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                )
            )

            validation_loss, validation_accuracy = (
                evaluate(
                    model=model,
                    data_loader=validation_loader,
                    criterion=criterion,
                    device=device,
                )
            )

            current_learning_rate = (
                optimizer.param_groups[0]["lr"]
            )

            history["epoch"].append(
                global_epoch
            )

            history["stage"].append(stage)

            history["learning_rate"].append(
                current_learning_rate
            )

            history["train_loss"].append(
                train_loss
            )

            history["train_accuracy"].append(
                train_accuracy
            )

            history["validation_loss"].append(
                validation_loss
            )

            history["validation_accuracy"].append(
                validation_accuracy
            )

            print(
                f"Stage {stage} | "
                f"Epoch {global_epoch:03d} | "
                f"lr: {current_learning_rate:.6f} | "
                f"train loss: {train_loss:.4f} | "
                f"train accuracy: {train_accuracy:.4f} | "
                f"validation loss: {validation_loss:.4f} | "
                f"validation accuracy: {validation_accuracy:.4f}"
            )

            is_better = (
                validation_loss
                < best_validation_loss - min_delta
            )

            if is_better:
                best_validation_loss = (
                    validation_loss
                )

                best_validation_accuracy = (
                    validation_accuracy
                )

                best_epoch = global_epoch
                best_stage = stage
                bad_epochs = 0

                best_model_state = copy.deepcopy(
                    model.state_dict()
                )

                best_optimizer_state = copy.deepcopy(
                    optimizer.state_dict()
                )

            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                print(
                    f"Early stopping на этапе {stage}. "
                    f"Нет улучшения {patience} эпох."
                )
                break

    if best_model_state is None:
        raise RuntimeError(
            "Обучение не создало ни одного checkpoint."
        )

    # Восстанавливаем глобально лучшую модель
    model.load_state_dict(
        best_model_state
    )

    # Финальный checkpoint
    checkpoint = {
        "epoch": best_epoch,
        "stage": best_stage,
        "model_state_dict": best_model_state,
        "optimizer_state_dict": best_optimizer_state,
        "validation_loss": best_validation_loss,
        "validation_accuracy": (
            best_validation_accuracy
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "history": history,
    }

    if checkpoint_metadata is not None:
        checkpoint.update(
            checkpoint_metadata
        )

    if checkpoint_path is not None:
        torch.save(
            checkpoint,
            checkpoint_path,
        )

    if history_path is not None:
        history_path.write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print(
        "Обучение завершено."
    )
    print(
        "Лучшая эпоха:",
        best_epoch,
    )
    print(
        "Лучший этап:",
        best_stage,
    )
    print(
        "Минимальный Validation Loss:",
        best_validation_loss,
    )
    print(
        "Validation Accuracy на лучшей эпохе:",
        best_validation_accuracy,
    )

    return history