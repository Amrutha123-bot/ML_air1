"""Configure sys.path so that 'src' is importable from backend/tests/."""
import sys
from pathlib import Path

# Add backend/src to Python path so `from src.xxx import yyy` works
_BACKEND = Path(__file__).parent.resolve()
_SRC = _BACKEND / "src"

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
