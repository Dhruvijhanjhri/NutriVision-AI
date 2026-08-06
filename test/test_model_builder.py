import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.model_builder import build_mobilenet

model, base_model = build_mobilenet(num_classes=30)

model.summary()