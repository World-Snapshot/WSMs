from pathlib import Path

import pytest

from wsms.catalog import TaskCatalog


ROOT = Path(__file__).resolve().parents[2] / "WSM_v1"


def test_catalog_combines_all_three_families():
    catalog = TaskCatalog.load(ROOT)
    assert len(catalog) >= 1150
    assert {task.source for task in catalog} == {"dense", "sparse", "custom"}


def test_repeated_head_indices_are_qualified():
    catalog = TaskCatalog.load(ROOT)
    assert catalog.by_head("dense:3").head == "dense:000003"
    assert catalog.by_head("sparse:3").head == "sparse:000003"
    with pytest.raises(ValueError, match="Bare Head indices are ambiguous"):
        catalog.by_head("3")


def test_resolve_dense_mapping():
    task = TaskCatalog.load(ROOT).resolve(inputs=["0-2"], output="36-38")
    assert task.head == "dense:000003"


def test_resolve_custom_mapping():
    task = TaskCatalog.load(ROOT).resolve(
        inputs=["0-2"], output="Universal_Pose"
    )
    assert task.head == "custom:000000"


def test_comma_containing_modality_is_one_token():
    tasks = TaskCatalog.load(ROOT).list(source="sparse", output="72-87,90-93")
    assert tasks
    assert tasks[0].output == "72-87,90-93"
