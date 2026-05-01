"""Make per-service packages importable for pytest without package collisions."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for p in [ROOT, ROOT / "command-api", ROOT / "fusion-engine", ROOT / "edge-agent"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
