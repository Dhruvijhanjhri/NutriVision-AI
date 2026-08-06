import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import create_directory

print("=" * 50)

create_directory("sample_output")

print("Utility Functions Working Successfully")

print("=" * 50)