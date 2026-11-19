from __future__ import annotations

import inspect
from collections import namedtuple

import numpy as np


def apply_legacy_smpl_compat() -> None:
    """Compatibility shims for old SMPL/chumpy pickles on modern Python/Numpy."""
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec  # type: ignore[attr-defined]

    # chumpy still references removed numpy scalar aliases.
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
