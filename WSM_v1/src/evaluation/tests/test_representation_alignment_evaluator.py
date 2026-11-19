from __future__ import annotations

import unittest

import torch

from src.evaluation.internal.representation_alignment_evaluator import (
    TRANSFORM_NAMES,
    apply_representation_transform,
    audit_representation,
    invert_representation_transform,
    training_timestep_quadrature,
)


class RepresentationAlignmentTransformTest(unittest.TestCase):
    def test_all_transforms_are_invertible_and_norm_preserving(self) -> None:
        values = torch.linspace(-2.0, 3.0, steps=2 * 12 * 128, dtype=torch.float64)
        native = values.reshape(2, 12, 128)
        for name in TRANSFORM_NAMES:
            with self.subTest(name=name):
                transformed = apply_representation_transform(native, name)
                restored = invert_representation_transform(transformed, name)
                self.assertTrue(torch.allclose(restored, native, atol=1e-12, rtol=1e-12))
                self.assertTrue(
                    torch.allclose(transformed.norm(), native.norm(), atol=1e-12, rtol=1e-12)
                )
                roundtrip, norm_change = audit_representation(native, name)
                self.assertLessEqual(roundtrip, 1e-12)
                self.assertLessEqual(norm_change, 1e-12)

    def test_transform_changes_coordinates(self) -> None:
        native = torch.arange(1 * 6 * 128, dtype=torch.float64).reshape(1, 6, 128)
        for name in TRANSFORM_NAMES[1:]:
            with self.subTest(name=name):
                transformed = apply_representation_transform(native, name)
                self.assertFalse(torch.equal(transformed, native))

    def test_channel_cycles_preserve_tokens_and_subpositions(self) -> None:
        # Each value records (token, channel, subposition).  A valid transform
        # may move channels but must never move tokens or the four 2x2 slots.
        native = torch.empty((1, 3, 32, 4), dtype=torch.int64)
        for token in range(3):
            for channel in range(32):
                for subposition in range(4):
                    native[0, token, channel, subposition] = (
                        token * 10000 + channel * 10 + subposition
                    )
        packed = native.reshape(1, 3, 128)
        for name in TRANSFORM_NAMES[1:]:
            with self.subTest(name=name):
                transformed = apply_representation_transform(packed, name).reshape(1, 3, 32, 4)
                self.assertTrue(torch.equal(transformed // 10000, native // 10000))
                self.assertTrue(
                    torch.equal(transformed.remainder(10), native.remainder(10))
                )

    def test_training_timestep_quadrature_matches_default_mixture(self) -> None:
        schedule = training_timestep_quadrature(7)
        self.assertAlmostEqual(sum(weight for _timestep, weight in schedule), 1.0)
        self.assertAlmostEqual(schedule[-1][0], 0.999)
        self.assertAlmostEqual(schedule[-1][1], 0.5)
        base_weight = sum(weight for _timestep, weight in schedule[:-1])
        self.assertAlmostEqual(base_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
