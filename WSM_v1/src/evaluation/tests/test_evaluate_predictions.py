from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.external.evaluate_predictions import (
    ContractError,
    depth_metrics,
    normal_metrics,
    paper_metric_names,
    resize_vector_field,
    select_metrics,
)


NORMAL_RANGES = np.asarray(
    [
        [-1.2443599700927734, 1.1317600011825562],
        [-1.3442360162734985, 1.3391599655151367],
        [-1.4413199424743652, 1.495519995689392],
    ],
    dtype=np.float64,
)


class NormalMetricContractTest(unittest.TestCase):
    def evaluate(self, prediction_values: np.ndarray, encoding: str) -> dict[str, float]:
        normal = np.asarray([0.2, -0.4, np.sqrt(0.8)], dtype=np.float64)
        target = np.broadcast_to(normal, (3, 5, 3)).copy()
        prediction = np.broadcast_to(prediction_values, (6, 10, 3)).copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "target.npy", target)
            np.save(root / "prediction.npy", prediction)
            sample = {
                "sample_id": "normal-contract",
                "target_path": "target.npy",
                "target_normal_encoding": "xyz",
            }
            prediction_record = {
                "prediction_path": "prediction.npy",
                "normal_encoding": encoding,
            }
            if encoding.startswith("wsg_"):
                prediction_record["normal_channel_ranges"] = NORMAL_RANGES.tolist()
            return normal_metrics(sample, prediction_record, root, {})

    def test_rgb_and_wsg_encodings_recover_the_same_physical_normal(self) -> None:
        normal = np.asarray([0.2, -0.4, np.sqrt(0.8)], dtype=np.float64)
        low = NORMAL_RANGES[:, 0]
        span = NORMAL_RANGES[:, 1] - low
        encodings = {
            "rgb_01": (normal + 1.0) / 2.0,
            "wsg_linear_01": (normal - low) / span,
            "wsg_log_01": np.log1p(normal - low) / np.log1p(span),
        }
        for encoding, values in encodings.items():
            with self.subTest(encoding=encoding):
                metrics = self.evaluate(values, encoding)
                self.assertLess(metrics["normal_mean_deg"], 1e-5)
                self.assertEqual(metrics["normal_within_11_25"], 1.0)

    def test_vector_resize_does_not_quantize_small_components(self) -> None:
        values = np.full((2, 2, 3), [0.0015, 0.5, 0.8], dtype=np.float64)
        resized = resize_vector_field(values, (7, 9))
        np.testing.assert_allclose(resized[..., 0], 0.0015, atol=1e-7)

    def test_zero_vectors_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "target.npy", np.zeros((2, 2, 3), dtype=np.float32))
            np.save(root / "prediction.npy", np.zeros((2, 2, 3), dtype=np.float32))
            with self.assertRaises(ContractError):
                normal_metrics(
                    {
                        "sample_id": "invalid-normal",
                        "target_path": "target.npy",
                        "target_normal_encoding": "xyz",
                    },
                    {
                        "prediction_path": "prediction.npy",
                        "normal_encoding": "xyz",
                    },
                    root,
                    {},
                )


class PaperMetricProfileTest(unittest.TestCase):
    def test_depth_profile_matches_paper_columns(self) -> None:
        self.assertEqual(
            paper_metric_names("depth_estimation"),
            {
                "unit_mae", "unit_rmse", "unit_acc_005", "unit_acc_010",
                "unit_acc_020", "unit_psnr_db", "unit_pearson", "unit_spearman",
            },
        )

    def test_paper_profile_filters_diagnostic_metrics(self) -> None:
        metrics = {
            "edge_accuracy": 0.9,
            "edge_mcc": 0.8,
            "edge_false_positive_rate": 0.1,
        }
        self.assertEqual(
            select_metrics(metrics, "any_to_any", "edge", False),
            {"edge_accuracy": 0.9, "edge_false_positive_rate": 0.1},
        )
        self.assertEqual(
            select_metrics(metrics, "any_to_any", "edge", True), metrics
        )

    def test_default_depth_computation_is_limited_to_paper_metrics(self) -> None:
        target = np.linspace(0.1, 1.0, 16, dtype=np.float32).reshape(4, 4)
        prediction = target * 0.8 + 0.1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "target.npy", target)
            np.save(root / "prediction.npy", prediction)
            sample = {"sample_id": "depth", "target_path": "target.npy"}
            predicted = {"prediction_path": "prediction.npy"}
            paper = depth_metrics(sample, predicted, root, {}, False)
            extended = depth_metrics(sample, predicted, root, {}, True)

        self.assertEqual(set(paper), paper_metric_names("depth_estimation"))
        self.assertIn("unit_psnr_db", paper)
        self.assertNotIn("abs_rel", paper)
        self.assertIn("abs_rel", extended)
        self.assertIn("median_abs_rel", extended)

    def test_default_row_removes_diagnostic_columns(self) -> None:
        from src.evaluation.external.evaluate_predictions import select_paper_row
        row = {
            "task": "semantic_estimation", "target_type": "semantic",
            "shared_miou": 0.8, "boundary_f1": 0.7, "iou_class_1": 0.6,
        }
        self.assertEqual(select_paper_row(row), {
            "task": "semantic_estimation", "target_type": "semantic",
            "shared_miou": 0.8,
        })

    def test_unknown_directed_task_fails_loudly(self) -> None:
        with self.assertRaises(ContractError):
            paper_metric_names("any_to_any", "flow")


if __name__ == "__main__":
    unittest.main()
