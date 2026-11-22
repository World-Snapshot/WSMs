from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import wsms.hub as hub
from wsms.container import HEADER_SIZE, WSMContainer, build_container
from wsms.catalog import WSMTask
from wsms.hub import resolve_artifacts


def _build_fixture(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    payloads = {
        "backbone/config.json": b'{"model":"tiny"}\n',
        "tunnelingnets/dense/000001/tunnelingnet.safetensors": b"one" * 7000,
        "tunnelingnets/dense/000002/tunnelingnet.safetensors": b"two" * 9000,
    }
    sources = tmp_path / "sources"
    members = []
    for name, payload in payloads.items():
        path = sources / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        members.append((name, path))
    output = tmp_path / "WSM_v1.wsm"
    build_container(
        output,
        members,
        model_index={
            "tunnelingnet_count": 2,
            "tasks": [
                {
                    "tunnelingnet_id": "dense:000001",
                    "checkpoint": members[1][0],
                    "artifacts": [members[1][0]],
                },
                {
                    "tunnelingnet_id": "dense:000002",
                    "checkpoint": members[2][0],
                    "artifacts": [members[2][0]],
                },
            ],
            "backbone": {"artifacts": [members[0][0]]},
        },
        alignment=4096,
    )
    return output, payloads


def test_single_file_header_index_and_selective_materialization(tmp_path: Path) -> None:
    output, payloads = _build_fixture(tmp_path)
    container = WSMContainer.open(output)

    assert output.read_bytes()[:8] == b"WSMFILE1"
    assert HEADER_SIZE == 64
    assert container.model["tunnelingnet_count"] == 2
    assert container.task_entry("dense:000002")["tunnelingnet_id"] == "dense:000002"

    selected = [
        "backbone/config.json",
        "tunnelingnets/dense/000002/tunnelingnet.safetensors",
    ]
    cache = container.materialize(selected, tmp_path / "cache")
    for name in selected:
        assert (cache / name).read_bytes() == payloads[name]
    assert not (cache / "tunnelingnets/dense/000001/tunnelingnet.safetensors").exists()


def test_local_materialization_avoids_whole_member_read(tmp_path: Path) -> None:
    output, payloads = _build_fixture(tmp_path)
    container = WSMContainer.open(output)
    calls: list[int] = []
    original = container.source.read

    def observed(offset: int, length: int) -> bytes:
        calls.append(length)
        return original(offset, length)

    container.source.read = observed  # type: ignore[method-assign]
    target = "tunnelingnets/dense/000002/tunnelingnet.safetensors"
    cache = container.materialize([target], tmp_path / "cache")
    assert (cache / target).read_bytes() == payloads[target]
    assert calls == []


def test_corrupt_index_is_rejected(tmp_path: Path) -> None:
    output, _ = _build_fixture(tmp_path)
    with output.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 0x01]))
    with pytest.raises(ValueError, match="index checksum"):
        WSMContainer.open(output)


def test_member_checksums_are_recorded(tmp_path: Path) -> None:
    output, payloads = _build_fixture(tmp_path)
    container = WSMContainer.open(output)
    for name, payload in payloads.items():
        assert container.member(name)["sha256"] == hashlib.sha256(payload).hexdigest()
    assert container.verify() == len(payloads)


def test_public_resolver_uses_embedded_backbone_and_selected_task(tmp_path: Path) -> None:
    output, _ = _build_fixture(tmp_path)
    runtime = tmp_path / "runtime"
    (runtime / "configs").mkdir(parents=True)
    task = WSMTask(
        source="dense",
        head_index=2,
        task_type="estimation_or_reconstruction",
        inputs=("0-2",),
        output="32-33",
        checkpoint="legacy/path/is/not/used.safetensors",
        run_dir="legacy/path",
    )
    artifacts = resolve_artifacts(
        output,
        task,
        cache_dir=tmp_path / "resolved",
        runtime=runtime,
    )
    assert artifacts.checkpoint.name == "tunnelingnet.safetensors"
    assert artifacts.checkpoint.read_bytes() == b"two" * 9000
    assert artifacts.base_model == artifacts.model_root / "backbone"
    assert (artifacts.base_model / "config.json").is_file()
    assert not (
        artifacts.model_root
        / "tunnelingnets/dense/000001/tunnelingnet.safetensors"
    ).exists()


def test_remote_legacy_repository_falls_back_to_split_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "legacy-snapshot"
    checkpoint = model_root / "train_log/example/Channel_Head_000002/best.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"legacy")
    task = WSMTask(
        source="dense",
        head_index=2,
        task_type="estimation_or_reconstruction",
        inputs=("0-2",),
        output="32-33",
        checkpoint=str(checkpoint.relative_to(model_root)),
        run_dir="train_log/example/Channel_Head_000002",
    )
    monkeypatch.setattr(hub, "_remote_has_container", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        hub, "_snapshot_download", lambda **kwargs: str(model_root)
    )
    runtime = tmp_path / "runtime"
    (runtime / "configs").mkdir(parents=True)

    artifacts = resolve_artifacts(
        "organization/legacy-wsm",
        task,
        runtime=runtime,
        download_base_model=False,
    )

    assert artifacts.checkpoint == checkpoint
    assert artifacts.base_model is None
