import sys
from pathlib import Path

# Ensure the project root is on the import path so ``src`` can be imported.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
