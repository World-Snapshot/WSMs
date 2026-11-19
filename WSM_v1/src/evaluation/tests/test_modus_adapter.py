from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.external.adapters.modus_adapter import stage_modality_input


class ModusAdapterTest(unittest.TestCase):
    def test_non_png_input_is_transcoded_to_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "normal.jpg"
            staged = root / "normal.png"
            pixels = np.zeros((8, 9, 3), dtype=np.uint8)
            pixels[..., 0] = 64
            pixels[..., 1] = 128
            pixels[..., 2] = 192
            Image.fromarray(pixels).save(source)

            stage_modality_input(source, staged, "normal")

            self.assertEqual(staged.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with Image.open(staged) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (9, 8))

    def test_canny_staging_is_binary_rgb_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "edge.png"
            staged = root / "staged.png"
            Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(source)
            stage_modality_input(source, staged, "canny")
            with Image.open(staged) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(set(np.asarray(image).reshape(-1)), {0, 255})


if __name__ == "__main__":
    unittest.main()
