"""Structured outputs returned by WSM pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from diffusers.utils import BaseOutput
from PIL import Image


@dataclass
class WSMPipelineOutput(BaseOutput):
    """In-memory result plus optional persistent inference artifacts."""

    images: list[Image.Image]
    wsg: np.ndarray | None = None
    task_id: str = ""
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
