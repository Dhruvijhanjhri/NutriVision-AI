import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.callbacks import get_callbacks

callbacks = get_callbacks()

print("=" * 50)
print("Callbacks Created Successfully")
print("=" * 50)

for cb in callbacks:
    print(type(cb).__name__)