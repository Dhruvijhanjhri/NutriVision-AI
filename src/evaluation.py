from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import OUTPUT_DIR


def evaluate_model(model, test_ds):

    predictions = model.predict(test_ds, verbose=1)

    y_pred = np.argmax(predictions, axis=1)

    y_true = np.concatenate(
        [labels.numpy() for _, labels in test_ds],
        axis=0
    )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted"
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted"
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="weighted"
        )
    }

    return metrics, y_true, y_pred


def print_metrics(metrics):

    print("=" * 50)

    print("Model Evaluation")

    print("=" * 50)

    for key, value in metrics.items():
        print(f"{key:12}: {value:.4f}")


def generate_classification_report(
    y_true,
    y_pred,
    class_names
):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print(report)

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(16, 14))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.show()


def plot_training_history(history):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "accuracy_curve.png",
        dpi=300
    )

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=300
    )

    plt.show()