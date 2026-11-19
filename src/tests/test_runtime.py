from pathlib import Path
from types import SimpleNamespace

import wsm.runtime as runtime


def test_runtime_preserves_huggingface_snapshot_path(tmp_path, monkeypatch):
    blob = tmp_path / "models--example" / "blobs" / "hash"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    checkpoint = (
        tmp_path
        / "models--example"
        / "snapshots"
        / "revision"
        / "train_log"
        / "Channel_Head_000003"
        / "run"
        / "ckpt"
        / "0001780"
        / "pytorch_lora_weights.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(blob)
    base_model = tmp_path / "base-model"
    base_model.mkdir()

    captured = {}

    class FakeSession:
        device = "cuda:0"

        def __init__(self, value, *, categorical_refiner):
            captured["checkpoint"] = value

    monkeypatch.setattr(
        runtime,
        "load_backend",
        lambda _: SimpleNamespace(ChannelHeadInferenceSession=FakeSession),
    )
    session = runtime.RuntimeSession(
        runtime_root=tmp_path,
        checkpoint=checkpoint,
        base_model=base_model,
        device="cuda:0",
        categorical_refiner="auto",
    )

    assert session.checkpoint == checkpoint.absolute()
    assert captured["checkpoint"] == checkpoint.absolute()
    assert session.checkpoint.parent.name == "0001780"
