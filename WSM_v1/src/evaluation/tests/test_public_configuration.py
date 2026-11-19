from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.evaluation.external.run_benchmark import PROJECT_ROOT, config_template


EXTERNAL_ROOT = PROJECT_ROOT / "src/evaluation/external"
EXAMPLE_CONFIG = EXTERNAL_ROOT / "benchmark_heldout24.example.json"
REGISTRY = EXTERNAL_ROOT / "official_registry.json"
RECOMMENDED_DENSE = PROJECT_ROOT / "configs/wsm_v1_recommended_dense_channel_heads.json"
SPARSE_CONFIG = PROJECT_ROOT / "configs/wsg_v1_modalities_sparse.json"
CUSTOM_CONFIG = PROJECT_ROOT / "configs/wsg_v1_modalities_custom.json"
SPARSE_LAUNCHER = PROJECT_ROOT / "scripts/launch_scripts/train_sparse_structure_models.sh"
SPARSE_RESULTS = (
    PROJECT_ROOT / "src/evaluation/docs/native_sparse_head_results_20260904.json"
)


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


class PublicConfigurationTest(unittest.TestCase):
    def test_example_is_portable_and_opt_in(self) -> None:
        payload = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        forbidden = ("/research/", "/common/users/", "/home/", "/ABSOLUTE/PATH")
        self.assertFalse(
            [value for value in strings(payload) if value.startswith(forbidden)]
        )
        self.assertFalse(
            [method["id"] for method in payload["methods"] if method.get("enabled")]
        )

    def test_init_paths_resolve_to_ignored_evaluation_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "benchmark.json"
            payload = config_template(destination)
            workspace = (destination.parent / payload["workspace"]).resolve()
            hf_home = (destination.parent / payload["hf_home"]).resolve()
        self.assertEqual(
            workspace,
            PROJECT_ROOT / "src/evaluation_output/external/wsm_external_heldout24",
        )
        self.assertEqual(
            hf_home,
            PROJECT_ROOT / "src/evaluation_output/external/_shared_hf_cache",
        )

    def test_repository_revisions_are_pinned(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        floating = []
        for method_id, method in registry["methods"].items():
            if method.get("repo_url") and not re.fullmatch(
                r"[0-9a-f]{40}", str(method.get("revision") or "")
            ):
                floating.append(method_id)
        self.assertEqual(floating, [])

    def test_compact_graphs_own_their_recommended_checkpoints(self) -> None:
        sparse = json.loads(SPARSE_CONFIG.read_text(encoding="utf-8"))["wsg_v1"]
        custom = json.loads(CUSTOM_CONFIG.read_text(encoding="utf-8"))["wsg_v1"]
        launcher = SPARSE_LAUNCHER.read_text(encoding="utf-8")
        match = re.search(
            r'^CHANNEL_HEADS=.*:-([^}]*)\}"',
            launcher,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        launched = {int(value) for value in match.group(1).split(",")}
        recommended = {
            int(name.rsplit("_", 1)[1])
            for name, entry in sparse.items()
            if entry.get("recommended_checkpoint")
        }
        self.assertEqual(recommended, launched)
        self.assertTrue(custom)
        self.assertTrue(all(entry.get("recommended_checkpoint") for entry in custom.values()))
        for family, entries in (("sparse", sparse), ("custom", custom)):
            for entry in entries.values():
                checkpoint = entry.get("recommended_checkpoint")
                if not checkpoint:
                    continue
                self.assertFalse(Path(checkpoint).is_absolute())
                self.assertTrue(checkpoint.startswith(f"train_log_{family}/"))

    def test_dense_recommendation_registry_contains_only_dense_heads(self) -> None:
        payload = json.loads(RECOMMENDED_DENSE.read_text(encoding="utf-8"))
        default_source = str(payload.get("head_source_default") or "dense")
        sources = {
            str(entry.get("head_source") or default_source)
            for entry in payload.get("entries") or []
        }
        self.assertEqual(sources, {"dense"})

    def test_sparse_results_follow_embedded_recommendations(self) -> None:
        sparse = json.loads(SPARSE_CONFIG.read_text(encoding="utf-8"))["wsg_v1"]
        results = json.loads(SPARSE_RESULTS.read_text(encoding="utf-8"))
        result_heads = results["heads"]
        recommended = {
            name: entry["recommended_checkpoint"]
            for name, entry in sparse.items()
            if entry.get("recommended_checkpoint")
        }
        self.assertEqual(set(result_heads), set(recommended))
        for name, checkpoint in recommended.items():
            self.assertEqual(result_heads[name]["checkpoint"], checkpoint)
            self.assertFalse(Path(result_heads[name]["checkpoint"]).is_absolute())
        self.assertIn(
            "pa_mpjpe_mm_mean",
            result_heads["Channel_Head_000000"]["metrics"],
        )
        self.assertNotIn(
            "mpjpe_mm",
            result_heads["Channel_Head_000000"]["metrics"],
        )


if __name__ == "__main__":
    unittest.main()
