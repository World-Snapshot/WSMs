#!/usr/bin/env python
"""Evaluation-only training entry for representation-alignment tests.

This module deliberately leaves the production trainer untouched.  It wraps
the production model at runtime and changes only the coordinate system of the
packed target latent.  Every optimizer, task recipe, attention path, loss,
timestep sampler, high-noise anchor, and checkpoint routine remains the
production implementation.

Usage is identical to ``scripts/train_TunnelingNet_Flux2_4B_wsg.py`` with one
additional argument::

    --representation-alignment native|vae-channel-cycle-01|...

The extra argument is removed before the production parser runs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_TunnelingNet_Flux2_4B_wsg as production  # noqa: E402
from src.evaluation.internal.representation_alignment_evaluator import (  # noqa: E402
    TRANSFORM_NAMES,
    apply_representation_transform,
    invert_representation_transform,
)


ACTIVE_REPRESENTATION = "native"
NATIVE_NORMAL_RENDER = False
SKIP_READER_VISUALS = False
ProductionModel = production.Flux2TunnelingNetModel


class RepresentationAlignmentModel(ProductionModel):
    """Production model with an evaluation-only target coordinate wrapper."""

    def _arm_target_transform(self, *, paired_noise: bool) -> None:
        self._alignment_target_pending = True
        self._alignment_noise_pending = bool(paired_noise)

    def normalize_and_pack(self, pipe: Any, latents):
        packed, ids = ProductionModel.normalize_and_pack(pipe, latents)
        if getattr(self, "_alignment_target_pending", False):
            packed = apply_representation_transform(packed, ACTIVE_REPRESENTATION)
            self._alignment_target_pending = False
        return packed, ids

    def sample_training_noise(
        self, target_clean, target_ids, timestep, generator, use_multires: bool = True
    ):
        noise = super().sample_training_noise(
            target_clean, target_ids, timestep, generator, use_multires=use_multires
        )
        if getattr(self, "_alignment_noise_pending", False):
            # A channel permutation leaves N(0,I) unchanged.  Applying the same
            # permutation merely pairs variants along the same random path; it
            # does not alter the production noise distribution.
            noise = apply_representation_transform(noise, ACTIVE_REPRESENTATION)
            self._alignment_noise_pending = False
        return noise

    def _compute_loss(self, *args, **kwargs):
        self._arm_target_transform(paired_noise=True)
        try:
            return super()._compute_loss(*args, **kwargs)
        finally:
            self._alignment_target_pending = False
            self._alignment_noise_pending = False

    def unpack_normalized_latents(self, packed, ids):
        native = invert_representation_transform(packed, ACTIVE_REPRESENTATION)
        return super().unpack_normalized_latents(native, ids)

    def run_preview(self, *args, **kwargs):
        self._arm_target_transform(paired_noise=False)
        try:
            return super().run_preview(*args, **kwargs)
        finally:
            self._alignment_target_pending = False
            self._alignment_noise_pending = False

    def run_inference(self, *args, **kwargs):
        self._arm_target_transform(paired_noise=False)
        try:
            return super().run_inference(*args, **kwargs)
        finally:
            self._alignment_target_pending = False
            self._alignment_noise_pending = False

    def save_wsg_reader_visualization(self, *args, **kwargs):
        if SKIP_READER_VISUALS:
            return None
        return super().save_wsg_reader_visualization(*args, **kwargs)

    def save_audio_mask_reader_comparison(self, *args, **kwargs):
        if SKIP_READER_VISUALS:
            return None
        return super().save_audio_mask_reader_comparison(*args, **kwargs)


def _extract_experiment_arguments() -> tuple[str, bool, bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--representation-alignment",
        choices=TRANSFORM_NAMES,
        default=os.environ.get("WSG_REPRESENTATION_ALIGNMENT", "native"),
    )
    parser.add_argument(
        "--representation-alignment-native-normal-render",
        action="store_true",
        help=(
            "Evaluation-only fixed XYZ-to-RGB rendering for surface normals. "
            "Unlike the reader preview, this uses no per-image contrast stretch, "
            "so prediction and target remain in one angular coordinate system."
        ),
    )
    parser.add_argument(
        "--representation-alignment-skip-reader-visuals",
        action="store_true",
        help="Skip auxiliary WSG Reader galleries during metric-only evaluation.",
    )
    experiment_args, production_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *production_args]
    return (
        str(experiment_args.representation_alignment),
        bool(experiment_args.representation_alignment_native_normal_render),
        bool(experiment_args.representation_alignment_skip_reader_visuals),
    )


def main() -> None:
    global ACTIVE_REPRESENTATION, NATIVE_NORMAL_RENDER, SKIP_READER_VISUALS
    (
        ACTIVE_REPRESENTATION,
        NATIVE_NORMAL_RENDER,
        SKIP_READER_VISUALS,
    ) = _extract_experiment_arguments()
    if NATIVE_NORMAL_RENDER:
        def fixed_signed_vector_rgb(values):
            return (values.clamp(-1, 1) + 1.0) * 127.5

        # This is a runtime-only patch inside the evaluation entry.  The
        # production trainer and its reader-oriented contrast rendering remain
        # unchanged on disk.
        production.render_signed_vector_rgb = fixed_signed_vector_rgb
    production.Flux2TunnelingNetModel = RepresentationAlignmentModel
    print(
        "[representation-alignment] evaluation wrapper active: "
        f"{ACTIVE_REPRESENTATION}; native_normal_render={NATIVE_NORMAL_RENDER}; "
        f"skip_reader_visuals={SKIP_READER_VISUALS}; "
        "production defaults unchanged",
        flush=True,
    )
    production.main()


if __name__ == "__main__":
    main()
