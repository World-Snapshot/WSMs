#!/usr/bin/env python3
"""Raw-input inference for compact sparse-structure Channel Heads."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCOREHMR_ROOT = PROJECT_ROOT / "third_party/ScoreHMR"
ADAPTER_ROOT = SCOREHMR_ROOT / "wsm_adapters"
for module_root in (PROJECT_ROOT, SCOREHMR_ROOT, ADAPTER_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))


def log(message: str) -> None:
    print(f"[Sparse Head Inference] {message}", flush=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint produced locally by the repository's sparse trainers."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        # Older pose checkpoints contain NumPy arrays in their metadata. They
        # are repository-produced artifacts, never user uploads.
        log(f"weights-only restore unavailable ({type(error).__name__}); loading trusted local checkpoint")
        return torch.load(path, map_location="cpu", weights_only=False)


def input_path(request: dict[str, Any], token: str) -> Path:
    for item in request.get("inputs") or []:
        if str(item.get("token")) == token:
            path = Path(str(item.get("path") or "")).resolve()
            if path.is_file():
                return path
            raise FileNotFoundError(f"Missing sparse input {token}: {path}")
    raise ValueError(f"Sparse Head requires input {token}")


def rgb_tensor(path: Path, size: int = 224) -> torch.Tensor:
    image = ImageOps.fit(
        Image.open(path).convert("RGB"),
        (size, size),
        method=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def draw_boxes(image_path: Path, boxes: np.ndarray, labels: list[str]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    colors = ((255, 64, 64), (64, 210, 255), (255, 210, 64), (120, 255, 140))
    for index, (box, label) in enumerate(zip(boxes, labels)):
        cx, cy, box_width, box_height = [float(value) for value in box]
        xyxy = (
            (cx - box_width / 2.0) * width,
            (cy - box_height / 2.0) * height,
            (cx + box_width / 2.0) * width,
            (cy + box_height / 2.0) * height,
        )
        color = colors[index % len(colors)]
        draw.rectangle(xyxy, outline=color, width=max(2, round(min(width, height) / 180)))
        draw.text((max(2, xyxy[0] + 3), max(2, xyxy[1] + 3)), label, fill=color)
    return image


def infer_bbox(request: dict[str, Any], checkpoint: Path) -> tuple[Image.Image, dict[str, Any]]:
    from wsm_adapters.train_wsv_bbox import PretrainedResNetCenterBoxHead, _select_prediction_indices

    image_path = input_path(request, "0-2")
    payload = load_checkpoint(checkpoint)
    max_slots = int(payload.get("max_slots", 10))
    model = PretrainedResNetCenterBoxHead(
        max_count=max_slots,
        load_pretrained_backbone=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    with torch.no_grad():
        logits, boxes, count_logits = model.forward_with_aux(rgb_tensor(image_path))
    scores = logits.sigmoid()[0].numpy()
    box_values = boxes[0].numpy()
    predicted_count = int(count_logits.argmax(dim=1)[0])
    keep = _select_prediction_indices(
        box_values,
        scores,
        predicted_count=predicted_count,
        ordered_slots=False,
    )
    selected = box_values[keep]
    labels = [f"person {rank + 1} · {scores[item]:.2f}" for rank, item in enumerate(keep)]
    return draw_boxes(image_path, selected, labels), {
        "task": "bbox",
        "count": len(keep),
        "boxes_cxcywh": selected.tolist(),
        "scores": [float(scores[item]) for item in keep],
    }


def pare_features(image_path: Path, work_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    from wsm_adapters.compat import apply_legacy_smpl_compat

    apply_legacy_smpl_compat()
    from score_hmr.configs import model_config
    from score_hmr.datasets.image_dataset import ImageDataset
    from score_hmr.models.model_utils import load_pare

    work_dir = work_dir.resolve()
    width, height = Image.open(image_path).size
    bbox_size = float(max(width, height))
    dataset_path = work_dir / "single_person_crop.npz"
    np.savez_compressed(
        dataset_path,
        imgname=np.asarray([str(image_path.resolve())]),
        center=np.asarray([[width / 2.0, height / 2.0]], dtype=np.float32),
        scale=np.asarray([[bbox_size, bbox_size]], dtype=np.float32),
    )
    # ScoreHMR addresses its released PARE/SMPL assets relative to its root.
    original_cwd = Path.cwd()
    try:
        os.chdir(SCOREHMR_ROOT)
        cfg = model_config()
        sample = ImageDataset(cfg, str(dataset_path), "/", train=False)[0]
        image_tensor = torch.from_numpy(sample["img"]).unsqueeze(0).to(device)
        pare = load_pare(cfg.SMPL).to(device).eval()
        with torch.no_grad():
            output = pare(image_tensor)
    finally:
        os.chdir(original_cwd)
    pose = output["pred_pose"]
    return {
        "features": torch.cat(
            (output["pose_feats"].reshape(1, -1), output["cam_shape_feats"].reshape(1, -1)),
            dim=1,
        ),
        "pare_pose_6d": pose[:, :, :, :2].transpose(2, 3).reshape(1, -1),
        "pare_cam": output["pred_cam"].reshape(1, -1),
        "pare_betas": output["pred_shape"].reshape(1, -1),
        "img": image_tensor,
        "pred_pose": pose,
        "pred_betas": output["pred_shape"],
        "pred_cam": output["pred_cam"],
    }


def pose_training_assets(task: str) -> tuple[Path, Path]:
    stem = f"wsv_structured_scorehmr_{task}_crop"
    dataset = PROJECT_ROOT / "data/sparse_structure/datasets" / f"{stem}_train.npz"
    cache = PROJECT_ROOT / "data/sparse_structure/cache/rgb_ch0_2_crop_pare" / f"{stem}_train.npz"
    if not dataset.is_file() or not cache.is_file():
        raise FileNotFoundError(f"Missing sparse {task} training-contract assets: {dataset}, {cache}")
    return dataset, cache


def draw_pose(
    image_path: Path,
    points: np.ndarray,
    confidence: np.ndarray,
    parents: np.ndarray,
    active_count: int,
    task: str,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    xy = np.clip(points[:, :2], 0.0, 1.0)
    valid = confidence >= 0.18
    line_color = (64, 220, 255) if task == "dwpose" else (255, 180, 64)
    for joint in range(min(active_count, len(points))):
        parent = int(parents[joint])
        if parent < 0 or parent >= active_count or not valid[joint] or not valid[parent]:
            continue
        start = (round(float(xy[parent, 0]) * (width - 1)), round(float(xy[parent, 1]) * (height - 1)))
        end = (round(float(xy[joint, 0]) * (width - 1)), round(float(xy[joint, 1]) * (height - 1)))
        draw.line((start, end), fill=line_color, width=max(2, round(min(width, height) / 220)))
    radius = max(2, round(min(width, height) / 150))
    for joint in range(min(active_count, len(points))):
        if not valid[joint]:
            continue
        x = round(float(xy[joint, 0]) * (width - 1))
        y = round(float(xy[joint, 1]) * (height - 1))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 72, 100), outline=(255, 255, 255))
    draw.rectangle((0, 0, min(width, 480), 28), fill=(0, 0, 0))
    draw.text((8, 7), f"{task}: {int(valid[:active_count].sum())} visible keypoints", fill=(255, 255, 255))
    return image


def infer_pose(
    request: dict[str, Any],
    checkpoint: Path,
    task: str,
    work_dir: Path,
    device: torch.device,
) -> tuple[Image.Image, dict[str, Any]]:
    from wsm_adapters.train_wsv_structure import KinematicPoseRegressor, StructureDataset, _kinematic_priors

    image_path = input_path(request, "0-2")
    dataset_path, cache_path = pose_training_assets(task)
    contract = StructureDataset(str(dataset_path), str(cache_path), "crop_pare")
    priors = _kinematic_priors(contract.target_raw, contract.target_mask, task, contract.max_keypoints)
    model = KinematicPoseRegressor(contract.features.shape[1], contract.max_keypoints, priors)
    payload = load_checkpoint(checkpoint)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    pare = pare_features(image_path, work_dir, device)
    batch = {key: pare[key] for key in ("features", "pare_pose_6d", "pare_cam", "pare_betas")}
    with torch.no_grad():
        prediction = model(batch)
    points = prediction["points"][0].float().cpu().numpy()
    confidence = prediction["score_logits"][0].sigmoid().float().cpu().numpy()
    return draw_pose(image_path, points, confidence, model.parents, model.active_count, task), {
        "task": task,
        "points": points[: model.active_count].tolist(),
        "confidence": confidence[: model.active_count].tolist(),
        "visible_threshold": 0.18,
    }


SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def draw_smpl_prediction(
    image_path: Path, joints: np.ndarray,
) -> Image.Image:
    source = Image.open(image_path).convert("RGB")
    width, height = source.size
    panel_width = max(360, width)
    canvas = Image.new("RGB", (width + panel_width, height), (18, 24, 35))
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    points = np.asarray(joints[:24, :2], dtype=np.float32)
    points -= points[[0]]
    span = np.ptp(points, axis=0)
    scale = 0.72 * min(panel_width / max(float(span[0]), 1e-4), height / max(float(span[1]), 1e-4))
    projected = np.empty_like(points)
    projected[:, 0] = width + panel_width * 0.5 + points[:, 0] * scale
    projected[:, 1] = height * 0.52 - points[:, 1] * scale
    for child, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        draw.line((*projected[parent], *projected[child]), fill=(82, 205, 255), width=max(2, height // 180))
    radius = max(3, height // 140)
    for x, y in projected:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 92, 126), outline=(255, 255, 255))
    draw.text((width + 18, 16), "Head 000000 · predicted SMPL structure", fill=(235, 242, 255))
    return canvas


def infer_smpl(
    request: dict[str, Any], checkpoint: Path, work_dir: Path, device: torch.device,
) -> tuple[Image.Image, dict[str, Any]]:
    from ema_pytorch import EMA
    from score_hmr.configs import model_config
    from score_hmr.models import FC, SMPL
    from score_hmr.utils import StandarizeImageFeatures, prepare_smpl_params
    from score_hmr import GaussianDiffusion

    image_path = input_path(request, "0-2")
    pare = pare_features(image_path, work_dir, device)
    payload = load_checkpoint(checkpoint)
    original_cwd = Path.cwd()
    try:
        os.chdir(SCOREHMR_ROOT)
        cfg = model_config()
        cfg.defrost()
        cfg.MODEL.USE_BETAS = True
        cfg.MODEL.DENOISING_MODEL.IMG_FEATS = "pare"
        cfg.EXTRA.LOAD_IMG_FEATS = "pare"
        cfg.EXTRA.LOAD_PREDICTIONS = "pare"
        cfg.freeze()
        model = FC(cfg).to(device)
        online = GaussianDiffusion(cfg, model, device=device).to(device)
        ema = EMA(online, beta=0.995, update_every=10)
        ema.load_state_dict(payload["ema"])
        diffusion = ema.ema_model.to(device).eval()
        standardize = StandarizeImageFeatures(
            backbone="pare", use_betas=True, device=device,
        )
        condition = standardize(pare["features"])
        seed = int(request.get("seed", 0))
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.no_grad():
            prediction = diffusion.sample({}, condition, batch_size=1)["x_0"]
            params = prepare_smpl_params(
                prediction,
                num_samples=1,
                use_betas=True,
                betas_min=diffusion.betas_min,
                betas_max=diffusion.betas_max,
            )
            smpl_cfg = {key.lower(): value for key, value in dict(cfg.SMPL).items()}
            smpl = SMPL(**smpl_cfg).to(device)
            smpl_output = smpl(**{key: value.float() for key, value in params.items()}, pose2rot=False)
    finally:
        os.chdir(original_cwd)
    joints = smpl_output.joints[0, :24].float().cpu().numpy()
    return draw_smpl_prediction(image_path, joints), {
        "task": "smpl",
        "joints_3d": joints.tolist(),
        "betas": params["betas"][0].float().cpu().tolist(),
        "global_orient": params["global_orient"][0].float().cpu().tolist(),
        "body_pose": params["body_pose"][0].float().cpu().tolist(),
    }


def audio_feature(audio_path: Path) -> torch.Tensor:
    from build_wsv_audio_dataset import AudioFeatureEncoder, _read_audio_file

    waveform, sample_rate = _read_audio_file(audio_path)
    encoder = AudioFeatureEncoder(
        "ast", 768, "MIT/ast-finetuned-audioset-10-10-0.4593", "cpu", 1, 10.0, 6.0
    )
    feature = encoder.encode_one(waveform, sample_rate=sample_rate, full_audio=False)
    return torch.from_numpy(np.asarray(feature, dtype=np.float32)).unsqueeze(0)


def infer_audio_localization(
    request: dict[str, Any], checkpoint: Path, device: torch.device,
) -> tuple[Image.Image, dict[str, Any]]:
    from wsm_adapters.train_wsv_audio import AudioCLIPJointMaskBoxLocalizationHead

    image_path = input_path(request, "0-2")
    payload = load_checkpoint(checkpoint)
    model = AudioCLIPJointMaskBoxLocalizationHead(
        audio_dim=int(payload.get("audio_dim", 768)),
        num_classes=26,
        init_checkpoint="",
        mask_init_checkpoint="",
        fusion_cross_weight=float(payload.get("fusion_cross_weight", 0.875)),
        mask_threshold=float(payload.get("mask_threshold", 0.57)),
        load_pretrained_clip=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    with torch.no_grad():
        box = model(
            rgb_tensor(image_path).to(device),
            audio_feature(input_path(request, "175-184")).to(device),
        )[0]
    box_np = box.float().cpu().numpy()[None]
    return draw_boxes(image_path, box_np, ["sounding source"]), {
        "task": "scene_audio_rgb_to_semantic_instance_localization",
        "box_cxcywh": box_np[0].tolist(),
    }


def infer_audio_category(
    request: dict[str, Any], checkpoint: Path, device: torch.device,
) -> tuple[Image.Image, dict[str, Any]]:
    from wsm_adapters.train_wsv_audio import BalancedPrototypeAudioCategoryHead, category_label

    payload = load_checkpoint(checkpoint)
    state = payload["model"]
    num_classes = int(state["prototypes"].shape[0] - 1)
    model = BalancedPrototypeAudioCategoryHead(
        int(payload.get("audio_dim", 768)), num_classes, torch.ones(num_classes + 1)
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    with torch.no_grad():
        probabilities = model(audio_feature(input_path(request, "175-184")).to(device)).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(5, probabilities.numel()))
    labels = [(int(index), category_label(int(index)), float(value)) for value, index in zip(values, indices)]
    canvas = Image.new("RGB", (720, 360), (18, 24, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "Predicted sound-source category", fill=(230, 240, 255))
    for rank, (index, label, probability) in enumerate(labels, start=1):
        y = 65 + 52 * (rank - 1)
        draw.text((35, y), f"{rank}. {label}  [{index}]", fill=(255, 255, 255))
        draw.rectangle((330, y, 330 + round(300 * probability), y + 24), fill=(70, 160, 240))
        draw.text((690, y), f"{probability:.3f}", anchor="ra", fill=(220, 230, 245))
    return canvas, {
        "task": "scene_audio_to_semantic_category",
        "topk": [
            {"index": index, "label": label, "probability": probability}
            for index, label, probability in labels
        ],
    }


def run(request: dict[str, Any], output_path: Path) -> dict[str, Any]:
    checkpoint = Path(str(request.get("checkpoint") or "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Sparse checkpoint does not exist: {checkpoint}")
    adapter = str(request.get("adapter_task") or "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"adapter={adapter} checkpoint={checkpoint} device={device}")
    if adapter == "smpl":
        image, native = infer_smpl(request, checkpoint, output_path.parent, device)
    elif adapter == "bbox":
        image, native = infer_bbox(request, checkpoint)
    elif adapter in {"dwpose", "mediapipe"}:
        image, native = infer_pose(request, checkpoint, adapter, output_path.parent, device)
    elif adapter == "scene_audio_rgb_to_semantic_instance_localization":
        image, native = infer_audio_localization(request, checkpoint, device)
    elif adapter == "scene_audio_to_semantic_category":
        image, native = infer_audio_category(request, checkpoint, device)
    else:
        raise NotImplementedError(f"Sparse adapter {adapter!r} has no raw-input inference implementation")
    image.save(output_path)
    native_path = output_path.parent / "native_prediction.json"
    native_path.write_text(json.dumps(native, indent=2), encoding="utf-8")
    manifest = {
        "outputs": [{
            "label": "Sparse adapter output",
            "path": str(output_path.resolve()),
            "filename": output_path.name,
            "selected": True,
        }],
        "native_prediction": str(native_path.resolve()),
    }
    (output_path.parent / "result_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"saved {output_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="ControlWorld request JSON")
    parser.add_argument("--output", required=True, help="Output visualization path")
    args = parser.parse_args()
    run(json.loads(Path(args.request).read_text(encoding="utf-8")), Path(args.output))


if __name__ == "__main__":
    main()
