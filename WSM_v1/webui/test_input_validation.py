from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from webui.input_validation import inspect_tensor_input


class InputValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "modalities.json"
        self.config.write_text(json.dumps({
            "normalization": {"channel_ranges": {
                "0": [0.0, 255.0], "1": [0.0, 255.0], "2": [0.0, 255.0],
                "34": [0.5, 70.0], "35": [0.0, 1.0],
                "36": [-1.5, 1.5], "37": [-1.5, 1.5], "38": [-1.5, 1.5],
                "67": [0.0, 255.0], "68": [0.0, 255.0], "69": [0.0, 255.0],
            }},
            "modality_specs": {"Universal_Pose": {
                "storage": "image_like", "channels": 3, "layout": "HWC",
                "value_range": [0.0, 255.0], "encoding": "rgb_0_255_to_minus1_1",
            }},
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save(self, name: str, array: np.ndarray, **metadata) -> Path:
        path = self.root / name
        np.savez(path, wsg_data=array, **metadata)
        return path

    def test_exact_scoped_tensor_is_green(self) -> None:
        path = self.save("depth.npz", np.ones((32, 32, 2), np.float32),
                         modality_id="34-35", channels=np.array([34, 35]))
        report = inspect_tensor_input(path, "34-35", self.config)
        self.assertEqual(report["severity"], "green")
        self.assertTrue(report["runnable"])

    def test_rgb_channels_are_scoped_not_full_wsg(self) -> None:
        path = self.save("rgb.npz", np.full((32, 32, 3), 128, np.uint8),
                         modality_id="0-2", channels=np.array([0, 1, 2]))
        report = inspect_tensor_input(path, "0-2", self.config)
        self.assertEqual(report["severity"], "green")
        self.assertEqual(report["contract"]["storage"], "modality_scoped")

    def test_inferred_scoped_tensor_is_orange(self) -> None:
        path = self.save("depth.npz", np.ones((32, 32, 2), np.float32))
        report = inspect_tensor_input(path, "34-35", self.config)
        self.assertEqual(report["severity"], "orange")

    def test_visualized_normal_is_red_but_retained(self) -> None:
        path = self.save("normal.npz", np.full((32, 32, 3), 128, np.uint8))
        report = inspect_tensor_input(path, "36-38", self.config)
        self.assertEqual(report["severity"], "red")
        self.assertTrue(report["accepted"])
        self.assertTrue(report["runnable"])

    def test_image_like_pose_is_not_mistaken_for_preview(self) -> None:
        path = self.save("pose.npz", np.full((32, 32, 3), 128, np.uint8),
                         modality_id="67-69", channels=np.array([67, 68, 69]))
        report = inspect_tensor_input(path, "67-69", self.config)
        self.assertEqual(report["severity"], "green")

    def test_nan_is_red_and_not_runnable(self) -> None:
        array = np.ones((32, 32, 2), np.float32)
        array[0, 0, 0] = np.nan
        path = self.save("nan.npz", array, modality_id="34-35",
                         channels=np.array([34, 35]))
        report = inspect_tensor_input(path, "34-35", self.config)
        self.assertEqual(report["severity"], "red")
        self.assertFalse(report["runnable"])

    def test_corrupt_contract_is_retained_red(self) -> None:
        path = self.save("wrong.npz", np.ones((32, 32, 4), np.float32))
        report = inspect_tensor_input(path, "34-35", self.config)
        self.assertEqual(report["severity"], "red")
        self.assertTrue(report["accepted"])
        self.assertFalse(report["runnable"])

    def test_string_modality_uses_explicit_contract(self) -> None:
        path = self.save("universal.npz", np.full((32, 32, 3), 128, np.uint8),
                         modality_id="Universal_Pose")
        report = inspect_tensor_input(path, "Universal_Pose", self.config)
        self.assertEqual(report["severity"], "green")

    def test_embedded_cache_preprocessing_contract_is_preserved(self) -> None:
        metadata = json.dumps({
            "training_normalization_mode": "log",
            "training_channel_ranges": {"34": [0.25, 80.0], "35": [0.0, 1.0]},
        })
        path = self.save(
            "cached_depth.npz", np.ones((32, 32, 2), np.float32),
            modality_id="34-35", channels=np.array([34, 35]),
            metadata_json=np.asarray(metadata),
        )
        report = inspect_tensor_input(path, "34-35", self.config)
        self.assertEqual(report["contract"]["training_normalization_mode"], "log")
        self.assertEqual(
            report["contract"]["training_channel_ranges"],
            {34: [0.25, 80.0], 35: [0.0, 1.0]},
        )


if __name__ == "__main__":
    unittest.main()
