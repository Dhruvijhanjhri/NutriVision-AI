from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from src.config import MODEL_DIR


def get_callbacks():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=MODEL_DIR / "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )

    csv_logger = CSVLogger(
        MODEL_DIR / "training_log.csv"
    )

    return [
        checkpoint,
        early_stop,
        reduce_lr,
        csv_logger
    ]