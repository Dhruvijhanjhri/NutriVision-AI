"""
config.py

Project configuration file for NutriVision-AI.
Contains all constants and paths used throughout the project.
"""
from pathlib import Path
import os

# Detect whether running on Google Colab
IS_COLAB = os.path.exists("/content")

if IS_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/NutriVision-AI")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "food30"

TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "validation"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Models
MODEL_DIR = PROJECT_ROOT / "models"

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

# ============================================================
# Dataset Configuration
# ============================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

CHANNELS = 3

NUM_CLASSES = 30

BATCH_SIZE = 32

SEED = 42

# ============================================================
# Training Configuration
# ============================================================

INITIAL_EPOCHS = 8

FINE_TUNE_EPOCHS = 5

INITIAL_LEARNING_RATE = 5e-4

FINE_TUNE_LEARNING_RATE = 1e-5

DROPOUT_RATE_1 = 0.40

DROPOUT_RATE_2 = 0.30

DENSE_UNITS = 256

# ============================================================
# Fine Tuning
# ============================================================

FINE_TUNE_LAST_N_LAYERS = 30

# ============================================================
# Callback Configuration
# ============================================================

EARLY_STOPPING_PATIENCE = 3

REDUCE_LR_PATIENCE = 2

REDUCE_LR_FACTOR = 0.2

# ============================================================
# Model Names
# ============================================================

MOBILENET_MODEL_NAME = "mobilenet_best.keras"

RESNET_MODEL_NAME = "resnet50_best.keras"

EFFICIENTNET_MODEL_NAME = "efficientnet_best.keras"

# ============================================================
# Visualization
# ============================================================

FIGURE_SIZE = (10, 6)

CONFUSION_MATRIX_SIZE = (12, 12)

# ============================================================
# Random Seed
# ============================================================

RANDOM_STATE = 42