import sys
from pathlib import Path

# Make the project root importable so tests can use `from src.x import ...`
sys.path.insert(0, str(Path(__file__).parent))
