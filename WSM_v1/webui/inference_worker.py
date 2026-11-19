"""Compatibility entry for integrations that still call the old WebUI worker."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.infer_TunnelingNet_Flux2_4B_wsg import main


if __name__ == "__main__":
    main()
