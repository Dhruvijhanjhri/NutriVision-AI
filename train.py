from pathlib import Path

from src.data_loader import load_datasets
from src.model_builder import build_mobilenet
from src.callbacks import get_callbacks
from src.utils import save_class_names
from src.evaluation import plot_training_history

from src.config import (
    MODEL_DIR,
    OUTPUT_DIR,
    INITIAL_EPOCHS
)


def main():
    print("\nProject Configuration")

    print("-" * 40)

    print(f"Training Images : {len(train_ds)} batches")

    print(f"Validation Images : {len(val_ds)} batches")

    print(f"Classes : {len(class_names)}")

    print(f"Epochs : {INITIAL_EPOCHS}")

    print("-" * 40)

    print("=" * 60)
    print("NutriVision-AI Training")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading Dataset...")

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("Done")

    print(f"Classes : {len(class_names)}")

    print("\nBuilding MobileNetV2...")

    model, base_model = build_mobilenet(
        num_classes=len(class_names)
    )

    print("Done")

    callbacks = get_callbacks()

    print("\nStarting Training...\n")

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=INITIAL_EPOCHS,

        callbacks=callbacks

    )

    print("\nTraining Completed")

    save_class_names(

        class_names,

        MODEL_DIR / "class_names.json"

    )

    plot_training_history(history)

    print("\nModel Saved Successfully")


if __name__ == "__main__":

    main()