import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.evaluation import print_metrics

dummy_metrics = {
    "Accuracy": 0.95,
    "Precision": 0.94,
    "Recall": 0.95,
    "F1 Score": 0.94
}

print_metrics(dummy_metrics)