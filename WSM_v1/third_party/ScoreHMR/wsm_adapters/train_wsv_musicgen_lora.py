from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import MusicgenForConditionalGeneration
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
try:
    hf_logging.disable_progress_bar()
except AttributeError:
    pass
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, TaskType, get_peft_model


class MetricLogger:
    def __init__(self, out_dir: Path) -> None:
        self.csv_path = out_dir / "metrics.csv"
        self.stats_path = out_dir / "stats.png"
        self.rows: list[tuple[int, float, float]] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self.csv_path.write_text("step,train_loss,val_loss\n")

    def append(self, step: int, train_loss: float, val_loss: float) -> None:
        self.rows.append((int(step), float(train_loss), float(val_loss)))
        with self.csv_path.open("a") as f:
            f.write(f"{int(step)},{float(train_loss):.8f},{float(val_loss):.8f}\n")
        self.plot()

    def plot(self) -> None:
        if not self.rows:
            return
        steps = [row[0] for row in self.rows]
        train = [row[1] for row in self.rows]
        val = [row[2] for row in self.rows]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, train, label="train_loss")
        plt.plot(steps, val, label="val_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.stats_path)
        plt.close()


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        audio = data.astype(np.float32) / max(1.0, float(np.iinfo(data.dtype).max))
    else:
        audio = data.astype(np.float32)
    return np.clip(audio, -1.0, 1.0), int(sr)


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32)
    if audio.size == 0:
        return np.zeros((target_sr,), dtype=np.float32)
    dst_len = max(1, int(round(audio.size * target_sr / source_sr)))
    src = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=dst_len, dtype=np.float32)
    return np.interp(dst, src, audio.astype(np.float32)).astype(np.float32)


def _audio_rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    return 20.0 * float(np.log10(max(rms, 1e-8)))


def _full_audio_path_from_image(img_path: str) -> Path | None:
    frame = Path(img_path)
    video_dir = frame.parent.parent
    wav_dir = video_dir / "WAVAudios"
    direct = wav_dir / f"{video_dir.name}.wav"
    if direct.exists():
        return direct
    matches = sorted(wav_dir.glob("*.wav")) if wav_dir.exists() else []
    return matches[0] if matches else None


class MusicGenImageAudioDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        image_size: int,
        sample_rate: int,
        audio_seconds: float,
        min_audio_rms_dbfs: float | None = None,
        max_samples: int | None = None,
    ) -> None:
        data = np.load(dataset_file, allow_pickle=True)
        imgname = data["imgname"].astype(str)
        rms = data["audio_rms_dbfs"].astype(np.float32) if "audio_rms_dbfs" in data else None
        cached_codes = data["audio_codes"].astype(np.int64) if "audio_codes" in data else None
        bbox = data["bbox"].astype(np.float32) if "bbox" in data else None
        audio_source = str(data["audio_source"].item()) if "audio_source" in data else ""
        frame_aligned_codes = cached_codes is not None and audio_source == "scene_audio_channels_175_184"
        keep_indices: list[int] = []
        self.audio_paths: list[str] = []
        seen: set[tuple[str, str]] = set()
        for index, img in enumerate(imgname):
            audio_path = None if frame_aligned_codes else _full_audio_path_from_image(img)
            if audio_path is None and cached_codes is None:
                continue
            rms_value = float(rms[index]) if rms is not None else None
            if min_audio_rms_dbfs is not None and rms_value is not None and rms_value < min_audio_rms_dbfs:
                continue
            if not frame_aligned_codes:
                # Full-audio targets are video-level; keep one representative per video/audio pair.
                key_source = str(audio_path) if audio_path is not None else str(data["wsv_path"][index])
                key = (key_source, Path(img).parent.parent.name)
                if key in seen:
                    continue
                seen.add(key)
            keep_indices.append(index)
            self.audio_paths.append(str(audio_path) if audio_path is not None else "")
            if max_samples is not None and len(keep_indices) >= max_samples:
                break
        if not keep_indices:
            raise RuntimeError(f"No MusicGen samples found in {dataset_file}")
        keep = np.asarray(keep_indices)
        self.imgname = imgname[keep]
        self.bbox = bbox[keep] if bbox is not None else None
        self.audio_codes = cached_codes[keep] if cached_codes is not None else None
        self.image_size = int(image_size)
        self.sample_rate = int(sample_rate)
        self.target_len = max(1, int(round(float(audio_seconds) * self.sample_rate)))
        print(
            f"[musicgen-data] {Path(dataset_file).name}: kept {len(self.imgname)} samples "
            f"from {len(imgname)} frame samples; source={audio_source or 'unknown'} "
            f"seconds={audio_seconds} sr={sample_rate} cached_codes={self.audio_codes is not None}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.imgname)

    def _crop_source(self, image: Image.Image, bbox: np.ndarray | None) -> Image.Image:
        if bbox is None or bbox.shape[0] < 4 or not np.all(np.isfinite(bbox[:4])):
            return image
        cx, cy, bw, bh = [float(v) for v in bbox[:4]]
        if bw <= 1e-4 or bh <= 1e-4:
            return image
        width, height = image.size
        x0 = int(round((cx - 0.5 * bw) * width))
        x1 = int(round((cx + 0.5 * bw) * width))
        y0 = int(round((cy - 0.5 * bh) * height))
        y1 = int(round((cy + 0.5 * bh) * height))
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        return image.crop((x0, y0, x1, y1))

    def _load_image(self, path: str, bbox: np.ndarray | None = None) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = self._crop_source(image, bbox).resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)))

    def _load_audio(self, path: str) -> tuple[torch.Tensor, float]:
        audio, sr = _read_audio(Path(path))
        audio = _resample(audio, sr, self.sample_rate)
        if audio.size < self.target_len:
            audio = np.pad(audio, (0, self.target_len - audio.size))
        else:
            audio = audio[: self.target_len]
        rms = _audio_rms_dbfs(audio)
        return torch.from_numpy(audio.astype(np.float32))[None, :], rms

    def __getitem__(self, index: int) -> dict:
        bbox = self.bbox[index] if self.bbox is not None else None
        item = {
            "image": self._load_image(self.imgname[index], bbox),
            "imgname": self.imgname[index],
            "audio_path": self.audio_paths[index],
        }
        if self.audio_codes is not None:
            item["audio_codes"] = torch.from_numpy(self.audio_codes[index].astype(np.int64))
            item["audio_rms_dbfs"] = torch.tensor(0.0, dtype=torch.float32)
            return item
        audio, rms = self._load_audio(self.audio_paths[index])
        item["audio"] = audio
        item["audio_rms_dbfs"] = torch.tensor(rms, dtype=torch.float32)
        return item


class ImagePrefixEncoder(nn.Module):
    def __init__(self, hidden_size: int, prefix_length: int) -> None:
        super().__init__()
        self.prefix_length = int(prefix_length)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 4 * 4, hidden_size * prefix_length),
            nn.LayerNorm(hidden_size * prefix_length),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        prefix = self.proj(self.backbone(image))
        return prefix.view(image.shape[0], self.prefix_length, -1)


def encode_musicgen_labels(model: MusicgenForConditionalGeneration, audio: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model.audio_encoder.encode(input_values=audio)
        codes = outputs.audio_codes
    frames, batch, codebooks, seq_len = codes.shape
    if frames != 1:
        raise RuntimeError(f"MusicGen audio encoder produced {frames} frames; expected 1. Use shorter/fixed audio.")
    if model.config.decoder.audio_channels == 2 and codebooks == model.decoder.num_codebooks // 2:
        codes = codes.repeat_interleave(2, dim=2)
    return codes[0].permute(0, 2, 1).contiguous().long()


def compute_loss(model, image_prefix, batch, device):
    prefix_dtype = next(image_prefix.parameters()).dtype
    images = batch["image"].to(device=device, dtype=prefix_dtype)
    if "audio_codes" in batch:
        labels = batch["audio_codes"].to(device).long()
    else:
        audio = batch["audio"].to(device)
        labels = encode_musicgen_labels(model, audio)
    encoder_hidden = image_prefix(images)
    attention_mask = torch.ones(encoder_hidden.shape[:2], dtype=torch.long, device=device)
    outputs = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    return outputs.loss


def compute_val_loss(model, image_prefix, loader, device, max_batches: int) -> float:
    model.eval()
    image_prefix.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            losses.append(float(compute_loss(model, image_prefix, batch, device).detach().cpu()))
    model.train()
    image_prefix.train()
    return float(np.mean(losses)) if losses else float("inf")


def save_audio(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    wavfile.write(path, sample_rate, clipped)
    peak = float(np.max(np.abs(clipped))) if clipped.size else 0.0
    if peak > 1e-6:
        wavfile.write(path.with_name(path.stem + "_listen.wav"), sample_rate, clipped / peak)


def decode_musicgen_labels(model: MusicgenForConditionalGeneration, labels: torch.Tensor) -> torch.Tensor | None:
    codes = labels.permute(0, 2, 1).unsqueeze(0).contiguous()
    try:
        decoded = model.audio_encoder.decode(audio_codes=codes, audio_scales=[None])
    except TypeError:
        try:
            decoded = model.audio_encoder.decode(codes, [None])
        except Exception:
            return None
    except Exception:
        return None
    audio = getattr(decoded, "audio_values", decoded)
    if isinstance(audio, tuple):
        audio = audio[0]
    return audio

def evaluate(model, image_prefix, loader, device, out_dir: Path, step: int, max_new_tokens: int) -> None:
    model.eval()
    image_prefix.eval()
    batch = next(iter(loader))
    prefix_dtype = next(image_prefix.parameters()).dtype
    images = batch["image"].to(device=device, dtype=prefix_dtype)[:2]
    with torch.no_grad():
        encoder_hidden = image_prefix(images)
        attention_mask = torch.ones(encoder_hidden.shape[:2], dtype=torch.long, device=device)
        generated = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=1.0,
        )
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    audio_np = generated.detach().float().cpu().numpy()
    if audio_np.ndim == 3:
        audio_np = audio_np[:, 0]
    for i in range(min(audio_np.shape[0], 2)):
        stem = out_dir / "visuals" / f"step_{step:06d}_sample_{i}"
        cond = batch["image"][i].detach().float().cpu().numpy()
        cond = np.transpose(cond, (1, 2, 0))
        cond = np.clip((cond * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(cond).save(stem.with_suffix(".png"))
        save_audio(stem.with_name(stem.name + "_pred.wav"), sample_rate, audio_np[i])
        if "audio" in batch:
            save_audio(stem.with_name(stem.name + "_gt.wav"), sample_rate, batch["audio"][i, 0].numpy())
        elif "audio_codes" in batch:
            gt_audio = decode_musicgen_labels(model, batch["audio_codes"][i : i + 1].to(device).long())
            if gt_audio is not None:
                gt_np = gt_audio.detach().float().cpu().numpy()
                if gt_np.ndim == 3:
                    gt_np = gt_np[:, 0]
                save_audio(stem.with_name(stem.name + "_gt.wav"), sample_rate, gt_np[0])
    model.train()
    image_prefix.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--model-name", default="facebook/musicgen-small")
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-size-val", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--checkpoint-freq", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-log-layout", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--audio-seconds", type=float, default=5.0)
    parser.add_argument("--min-audio-rms-dbfs", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--train-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-val-batches", type=int, default=8)
    parser.add_argument("--eval-new-tokens", type=int, default=128)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    print(f"[musicgen-{args.train_mode}] loading {args.model_name} dtype={dtype} device={device}", flush=True)
    model = MusicgenForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=dtype)
    decoder_bos = getattr(model.config.decoder, "bos_token_id", None)
    decoder_pad = getattr(model.config.decoder, "pad_token_id", None)
    decoder_start = decoder_bos if decoder_bos is not None else decoder_pad
    if getattr(model.config.decoder, "decoder_start_token_id", None) is None:
        model.config.decoder.decoder_start_token_id = decoder_start
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = model.config.decoder.decoder_start_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = decoder_pad if decoder_pad is not None else model.config.decoder_start_token_id
    model.to(device)
    model.freeze_audio_encoder()
    for param in model.text_encoder.parameters():
        param.requires_grad_(False)
    for param in model.audio_encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(False)

    if args.train_mode == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if name.startswith("base_model.model.text_encoder") or name.startswith("base_model.model.audio_encoder"):
                param.requires_grad_(False)
        model.print_trainable_parameters()
        base_model = model.base_model.model
    else:
        for param in model.decoder.parameters():
            param.requires_grad_(True)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100.0 * trainable_params / max(1, total_params):.4f}",
            flush=True,
        )
        base_model = model

    text_hidden = int(base_model.text_encoder.config.hidden_size)
    image_prefix = ImagePrefixEncoder(text_hidden, args.prefix_length).to(device=device, dtype=dtype)

    sample_rate = int(base_model.config.audio_encoder.sampling_rate)
    train_set = MusicGenImageAudioDataset(
        args.train_dataset, args.image_size, sample_rate, args.audio_seconds, args.min_audio_rms_dbfs, args.max_train_samples
    )
    val_set = MusicGenImageAudioDataset(
        args.val_dataset, args.image_size, sample_rate, args.audio_seconds, args.min_audio_rms_dbfs, args.max_val_samples
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size_val, shuffle=False, num_workers=args.num_workers, drop_last=False)

    output_root = Path(args.output_dir)
    ckpt_dir = output_root / "ckpt" if args.clean_log_layout else output_root / "checkpoints" / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "visuals").mkdir(parents=True, exist_ok=True)
    metrics = MetricLogger(output_root)
    (output_root / "resolved_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    trainable = [p for p in list(model.parameters()) + list(image_prefix.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    best_val = float("inf")
    step = 0
    progress = tqdm(total=args.total_steps, desc=f"train musicgen image-{args.train_mode}")
    while step < args.total_steps:
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(model, image_prefix, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            progress.update(1)
            progress.set_postfix(loss=float(loss.detach().cpu()))
            if step % args.log_freq == 0 or step == args.total_steps:
                val = compute_val_loss(model, image_prefix, val_loader, device, args.max_val_batches)
                metrics.append(step, float(loss.detach().cpu()), val)
                print(f"[musicgen-{args.train_mode}] step={step} train_loss={float(loss.detach().cpu()):.6f} val_loss={val:.6f}", flush=True)
                evaluate(model, image_prefix, val_loader, device, output_root, step, args.eval_new_tokens)
                if val < best_val:
                    best_val = val
                    best_dir = ckpt_dir / "best"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    best_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(best_dir / ("musicgen_lora" if args.train_mode == "lora" else "musicgen_full"))
                    torch.save({"step": step, "image_prefix": image_prefix.state_dict(), "best_val": best_val}, best_dir / "image_prefix.pt")
            if step % args.checkpoint_freq == 0 or step == args.total_steps:
                current_dir = ckpt_dir / "latest"
                if current_dir.exists():
                    shutil.rmtree(current_dir)
                current_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(current_dir / ("musicgen_lora" if args.train_mode == "lora" else "musicgen_full"))
                torch.save({"step": step, "image_prefix": image_prefix.state_dict(), "best_val": best_val}, current_dir / "image_prefix.pt")
            if step >= args.total_steps:
                break
    progress.close()
    print(f"[musicgen-{args.train_mode}] finished; best_val={best_val:.6f}; outputs under {output_root}", flush=True)


if __name__ == "__main__":
    main()
