from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from wsv_smpl_utils import frame_key_to_image_path, iter_wsv_files, split_train_val


AUDIO_CHANNELS = slice(175, 185)
SEMANTIC_CHANNEL = 195
MASK_VALUE = -1.0
MASK_SAVE_SIZE = 224


def resize_binary_mask(mask: np.ndarray, size: int = MASK_SAVE_SIZE) -> np.ndarray:
    mask = np.asarray(mask) > 0
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    if mask.shape == (size, size):
        return mask.astype(np.uint8)
    y_idx = np.linspace(0, mask.shape[0] - 1, size).round().astype(np.int64)
    x_idx = np.linspace(0, mask.shape[1] - 1, size).round().astype(np.int64)
    return mask[np.ix_(y_idx, x_idx)].astype(np.uint8)


class AudioFeatureEncoder:
    def __init__(
        self,
        encoder: str,
        audio_dim: int,
        model_name: str,
        device: str,
        batch_size: int,
        codec_seconds: float,
        codec_bandwidth: float,
    ) -> None:
        self.encoder = encoder
        self.audio_dim = audio_dim
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.codec_seconds = float(codec_seconds)
        self.codec_bandwidth = float(codec_bandwidth)
        self.codec_sample_rate = 24000
        self.codebook_size = 1024
        self.code_shape: tuple[int, int] | None = None
        self._feature_extractor: Any | None = None
        self._model: Any | None = None
        self.embedding_dim = audio_dim
        if encoder == "ast":
            self._load_ast()
        elif encoder == "encodec":
            self._load_encodec()
        elif encoder == "musicgen":
            self._load_musicgen()

    def _load_ast(self) -> None:
        from transformers import AutoModel

        # Do not use HuggingFace ASTFeatureExtractor here: in this environment it
        # imports torchaudio, whose CUDA build currently mismatches torch. We feed
        # AST with a local numpy/scipy log-mel implementation instead.
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        hidden_size = getattr(getattr(self._model, "config", None), "hidden_size", None)
        self.embedding_dim = int(hidden_size) if hidden_size is not None else 768
        print(f"[audio-encoder] loaded AST model={self.model_name} embedding_dim={self.embedding_dim} device={self.device}", flush=True)

    def _load_encodec(self) -> None:
        from transformers import EncodecModel

        self._model = EncodecModel.from_pretrained(self.model_name).to(self.device).eval()
        config = getattr(self._model, "config", None)
        self.codec_sample_rate = int(getattr(config, "sampling_rate", 24000))
        self.codebook_size = int(getattr(config, "codebook_size", 1024))
        print(
            f"[audio-encoder] loaded EnCodec model={self.model_name} sample_rate={self.codec_sample_rate} "
            f"codebook_size={self.codebook_size} bandwidth={self.codec_bandwidth} device={self.device}",
            flush=True,
        )

    def _load_musicgen(self) -> None:
        from transformers import MusicgenForConditionalGeneration

        self._model = MusicgenForConditionalGeneration.from_pretrained(self.model_name).to(self.device).eval()
        audio_config = getattr(getattr(self._model, "config", None), "audio_encoder", None)
        self.codec_sample_rate = int(getattr(audio_config, "sampling_rate", 32000))
        self.codebook_size = int(getattr(audio_config, "codebook_size", 2048))
        print(
            f"[audio-encoder] loaded MusicGen tokenizer model={self.model_name} sample_rate={self.codec_sample_rate} "
            f"codebook_size={self.codebook_size} seconds={self.codec_seconds} device={self.device}",
            flush=True,
        )


    @staticmethod
    def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
        def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
            return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)

        def mel_to_hz(mel: np.ndarray) -> np.ndarray:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            left, center, right = bins[m - 1], bins[m], bins[m + 1]
            if center > left:
                fb[m - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
            if right > center:
                fb[m - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
        return fb

    @classmethod
    def _ast_input(
        cls,
        audio: np.ndarray,
        source_sr: int = 8000,
        target_length: int = 1024,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        from scipy.signal import stft

        waveform = cls._resample(audio, source_sr, sample_rate)
        waveform = np.clip(waveform.astype(np.float32), -1.0, 1.0)
        if waveform.size < 400:
            waveform = np.pad(waveform, (0, 400 - waveform.size))
        _, _, spec = stft(
            waveform,
            fs=sample_rate,
            window="hann",
            nperseg=400,
            noverlap=240,
            nfft=512,
            boundary=None,
            padded=False,
        )
        power = (np.abs(spec) ** 2).astype(np.float32)
        mel_fb = cls._mel_filterbank(sample_rate, 512, 128, 20.0, 7600.0)
        mel = np.maximum(mel_fb @ power, 1e-10)
        log_mel = np.log(mel).T.astype(np.float32)
        if log_mel.shape[0] < target_length:
            log_mel = np.pad(log_mel, ((0, target_length - log_mel.shape[0]), (0, 0)))
        else:
            log_mel = log_mel[:target_length]
        # Defaults used by the public AST AudioSet feature extractor.
        return ((log_mel - (-4.2677393)) / (4.5689974 * 2.0)).astype(np.float32)

    @staticmethod
    def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        if source_sr == target_sr:
            return audio.astype(np.float32)
        if audio.size == 0:
            return np.zeros((target_sr,), dtype=np.float32)
        dst_len = max(1, int(round(audio.size * target_sr / source_sr)))
        src = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float32)
        dst = np.linspace(0.0, 1.0, num=dst_len, dtype=np.float32)
        return np.interp(dst, src, audio.astype(np.float32)).astype(np.float32)

    def encode_one(self, audio: np.ndarray, sample_rate: int = 8000, full_audio: bool = False) -> np.ndarray:
        if self.encoder == "raw":
            return _fixed_audio_feature(audio, self.audio_dim)
        return self.encode_batch([audio], [sample_rate], full_audio=full_audio)[0]

    def _encodec_feature(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        assert self._model is not None
        waveform = self._resample(audio, sample_rate, self.codec_sample_rate)
        waveform = np.clip(waveform.astype(np.float32), -1.0, 1.0)
        target_len = max(1, int(round(self.codec_seconds * self.codec_sample_rate)))
        if waveform.size < target_len:
            waveform = np.pad(waveform, (0, target_len - waveform.size))
        else:
            waveform = waveform[:target_len]
        inputs = torch.from_numpy(waveform).to(self.device).view(1, 1, -1)
        with torch.no_grad():
            encoded = self._model.encode(inputs, bandwidth=self.codec_bandwidth)
        codes = encoded.audio_codes.detach().cpu().numpy().astype(np.int64)
        # HF EnCodec returns (chunks, batch, codebooks, frames). With fixed,
        # unchunked inputs this is (1, 1, Q, T). Store normalized ids so the
        # lightweight visual-to-audio head can regress a fixed vector, then round
        # back to code ids for decoding during visualization.
        codes = codes.reshape(-1, codes.shape[-1])
        self.code_shape = (int(codes.shape[0]), int(codes.shape[1]))
        feature = codes.astype(np.float32) / float(self.codebook_size - 1)
        return feature.reshape(-1) * 2.0 - 1.0

    def _musicgen_codes(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        assert self._model is not None
        waveform = self._resample(audio, sample_rate, self.codec_sample_rate)
        waveform = np.clip(waveform.astype(np.float32), -1.0, 1.0)
        target_len = max(1, int(round(self.codec_seconds * self.codec_sample_rate)))
        if waveform.size < target_len:
            waveform = np.pad(waveform, (0, target_len - waveform.size))
        else:
            waveform = waveform[:target_len]
        inputs = torch.from_numpy(waveform).to(self.device).view(1, 1, -1)
        with torch.no_grad():
            encoded = self._model.audio_encoder.encode(input_values=inputs)
        codes = encoded.audio_codes.detach().cpu().numpy().astype(np.int64)
        if codes.shape[0] != 1 or codes.shape[1] != 1:
            raise RuntimeError(f"Unexpected MusicGen audio code shape {codes.shape}; expected (1, 1, Q, T)")
        codes = codes[0, 0].T.astype(np.int32)
        self.code_shape = (int(codes.shape[0]), int(codes.shape[1]))
        return codes

    def encode_batch(
        self,
        audios: list[np.ndarray],
        sample_rates: list[int] | None = None,
        full_audio: bool = False,
    ) -> np.ndarray:
        if self.encoder == "raw":
            return np.stack([_fixed_audio_feature(audio, self.audio_dim) for audio in audios], axis=0).astype(np.float32)
        sample_rates = sample_rates or [8000] * len(audios)
        if self.encoder == "encodec":
            features = [self._encodec_feature(audio, sr) for audio, sr in zip(audios, sample_rates)]
            return np.stack(features, axis=0).astype(np.float32)
        if self.encoder == "musicgen":
            codes = [self._musicgen_codes(audio, sr) for audio, sr in zip(audios, sample_rates)]
            code_array = np.stack(codes, axis=0).astype(np.int32)
            return code_array.reshape(code_array.shape[0], -1).astype(np.float32)
        if self.encoder != "ast":
            raise ValueError(f"Unsupported audio encoder: {self.encoder}")

        import torch

        assert self._model is not None
        sample_rates = sample_rates or [8000] * len(audios)
        if full_audio:
            encoded: list[np.ndarray] = []
            window = 16000 * 10
            for audio, sr in zip(audios, sample_rates):
                waveform = self._resample(audio, sr, 16000)
                if waveform.size <= window:
                    encoded.append(self.encode_batch([waveform], [16000], full_audio=False)[0])
                    continue
                chunks = [waveform[start : start + window] for start in range(0, waveform.size, window)]
                chunk_feat = self.encode_batch(chunks, [16000] * len(chunks), full_audio=False)
                encoded.append(chunk_feat.mean(axis=0).astype(np.float32))
            return np.stack(encoded, axis=0).astype(np.float32)

        features: list[np.ndarray] = []
        for start in range(0, len(audios), self.batch_size):
            chunk_audios = audios[start : start + self.batch_size]
            chunk_srs = sample_rates[start : start + self.batch_size]
            chunk = np.stack([self._ast_input(audio, source_sr=sr) for audio, sr in zip(chunk_audios, chunk_srs)], axis=0)
            inputs = torch.from_numpy(chunk).to(self.device)
            with torch.no_grad():
                outputs = self._model(input_values=inputs)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    pooled = outputs.last_hidden_state.mean(dim=1)
            features.append(pooled.detach().float().cpu().numpy())
        return np.concatenate(features, axis=0).astype(np.float32)


def _decode_audio_bytes(frame: np.ndarray, max_raw_bytes: int = 9000) -> tuple[np.ndarray | None, np.ndarray]:
    if frame.ndim != 3 or frame.shape[-1] < AUDIO_CHANNELS.stop:
        return None, np.zeros(frame.shape[:2], dtype=bool)
    audio_channels = frame[:, :, AUDIO_CHANNELS]
    audio_mask = np.any(audio_channels != MASK_VALUE, axis=2)
    ys, xs = np.where(audio_mask)
    if ys.size == 0:
        return None, audio_mask

    raw: list[int] = []
    for y, x in sorted(zip(ys.tolist(), xs.tolist())):
        for value in audio_channels[y, x]:
            if value == MASK_VALUE:
                continue
            byte_value = int((float(value) / 100.0 + 0.5) * 255.0)
            raw.append(int(np.clip(byte_value, 0, 255)))
            if len(raw) >= 14:
                length = raw[6] | (raw[7] << 8) | (raw[8] << 16) | (raw[9] << 24)
                if length == 7743:
                    length = 8000
                expected_total = 14 + max(0, min(length, max_raw_bytes))
                if len(raw) >= expected_total:
                    audio_uint8 = np.asarray(raw[14:expected_total], dtype=np.uint8)
                    if audio_uint8.size > 0:
                        return ((audio_uint8.astype(np.float32) - 128.0) / 127.0), audio_mask
            if len(raw) >= max_raw_bytes:
                break
        if len(raw) >= max_raw_bytes:
            break
    if len(raw) > 14:
        audio_uint8 = np.asarray(raw[14:], dtype=np.uint8)
        return ((audio_uint8.astype(np.float32) - 128.0) / 127.0), audio_mask
    return None, audio_mask


def _fixed_audio_feature(audio: np.ndarray, audio_dim: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((audio_dim,), dtype=np.float32)
    if audio.size == audio_dim:
        return audio.astype(np.float32)
    src = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=audio_dim, dtype=np.float32)
    return np.interp(dst, src, audio.astype(np.float32)).astype(np.float32)


def _video_dir_from_wsv_path(wsv_path: Path) -> Path:
    return wsv_path.parent.parent if wsv_path.parent.name.lower() == "wsv" else wsv_path.parent


def _full_audio_path_from_wsv_path(wsv_path: Path) -> Path | None:
    video_dir = _video_dir_from_wsv_path(wsv_path)
    candidates = [
        video_dir / "WAVAudios" / f"{video_dir.name}.wav",
        video_dir / "audio.wav",
        video_dir / "crop_original_video.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    wav_dir = video_dir / "WAVAudios"
    if wav_dir.exists():
        wavs = sorted(wav_dir.glob("*.wav"))
        if wavs:
            return wavs[0]
    return None


def _read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)
    original_dtype = data.dtype
    if np.issubdtype(original_dtype, np.unsignedinteger):
        info = np.iinfo(original_dtype)
        midpoint = 0.5 * float(info.max + info.min)
        waveform = (data.astype(np.float32) - midpoint) / max(1.0, midpoint)
    elif np.issubdtype(original_dtype, np.integer):
        waveform = data.astype(np.float32) / max(1.0, float(np.iinfo(original_dtype).max))
    else:
        waveform = data.astype(np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return np.clip(waveform, -1.0, 1.0), int(sample_rate)


def _audio_rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    return 20.0 * float(np.log10(max(rms, 1e-8)))


def _dominant_category(semantic: np.ndarray) -> int:
    valid = semantic[(semantic != MASK_VALUE) & (semantic > 0)]
    if valid.size == 0:
        return 0
    values, counts = np.unique(np.rint(valid).astype(np.int32), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    return np.asarray(
        [
            (x0 + 0.5 * bw) / max(1, width),
            (y0 + 0.5 * bh) / max(1, height),
            bw / max(1, width),
            bh / max(1, height),
        ],
        dtype=np.float32,
    )


def _sample_from_frame(frame: np.ndarray) -> dict | None:
    audio, audio_mask = _decode_audio_bytes(frame)
    if audio is None:
        return None
    if frame.ndim != 3 or frame.shape[-1] <= SEMANTIC_CHANNEL:
        return None
    semantic = frame[:, :, SEMANTIC_CHANNEL]
    category = _dominant_category(semantic)
    if category <= 0:
        return None

    # The injected scene audio is spatially written on the sound-source region.
    # Use that as the localization target; fall back to the semantic category
    # mask only when the audio mask is too small.
    source_mask = audio_mask
    bbox = _bbox_from_mask(source_mask)
    if bbox is None or float(bbox[2] * bbox[3]) < 1e-5:
        source_mask = np.rint(semantic).astype(np.int32) == category
        bbox = _bbox_from_mask(source_mask)
    if bbox is None:
        return None

    return {
        "audio_waveform": audio.astype(np.float32),
        "bbox": bbox,
        "mask": source_mask.astype(np.uint8),
        "category": category,
    }


def build_dataset(
    wsv_paths: list[Path],
    out_path: Path,
    frame_stride: int,
    max_frames_per_video: int | None,
    audio_encoder: AudioFeatureEncoder,
    audio_source: str,
    frame_samples_per_video: int | None,
    frame_sample_seed: int,
    min_audio_rms_dbfs: float | None,
) -> None:
    rows = {"imgname": [], "audio_waveform": [], "audio_sample_rate": [], "audio_rms_dbfs": [], "bbox": [], "mask": [], "category": [], "wsv_path": [], "frame_key": []}
    skipped_quiet = 0
    full_audio_cache: dict[Path, tuple[np.ndarray, int] | None] = {}
    rng = np.random.default_rng(frame_sample_seed)
    for wsv_path in tqdm(wsv_paths, desc=f"build {out_path.name}"):
        try:
            data = np.load(wsv_path)
        except Exception as exc:
            print(f"[skip] failed to open {wsv_path}: {exc}", flush=True)
            continue
        with data:
            keys = sorted(data.files)
            if frame_stride > 1:
                keys = keys[::frame_stride]
            if max_frames_per_video is not None:
                keys = keys[:max_frames_per_video]
            if audio_source == "full":
                full_audio = full_audio_cache.get(wsv_path)
                if wsv_path not in full_audio_cache:
                    audio_path = _full_audio_path_from_wsv_path(wsv_path)
                    if audio_path is None:
                        print(f"[skip] no full wav found for {wsv_path}", flush=True)
                        full_audio = None
                    else:
                        try:
                            full_audio = _read_audio_file(audio_path)
                        except Exception as exc:
                            print(f"[skip] failed to read full wav {audio_path}: {exc}", flush=True)
                            full_audio = None
                    full_audio_cache[wsv_path] = full_audio
                if full_audio is None:
                    continue
                full_audio_rms_dbfs = _audio_rms_dbfs(full_audio[0])
                if min_audio_rms_dbfs is not None and full_audio_rms_dbfs < min_audio_rms_dbfs:
                    skipped_quiet += 1
                    continue
            video_rows = {key: [] for key in rows}
            for key in keys:
                image_path = frame_key_to_image_path(wsv_path, key)
                if not image_path.exists():
                    continue
                sample = _sample_from_frame(data[key])
                if sample is None:
                    continue
                video_rows["imgname"].append(str(image_path))
                if audio_source == "full":
                    assert full_audio is not None
                    audio_waveform = full_audio[0]
                    audio_sample_rate = full_audio[1]
                    audio_rms_dbfs = full_audio_rms_dbfs
                else:
                    audio_waveform = sample["audio_waveform"]
                    audio_sample_rate = 8000
                    audio_rms_dbfs = _audio_rms_dbfs(audio_waveform)
                    if min_audio_rms_dbfs is not None and audio_rms_dbfs < min_audio_rms_dbfs:
                        skipped_quiet += 1
                        continue
                video_rows["audio_waveform"].append(audio_waveform)
                video_rows["audio_sample_rate"].append(audio_sample_rate)
                video_rows["audio_rms_dbfs"].append(audio_rms_dbfs)
                video_rows["bbox"].append(sample["bbox"])
                video_rows["mask"].append(sample["mask"])
                video_rows["category"].append(sample["category"])
                video_rows["wsv_path"].append(str(wsv_path))
                video_rows["frame_key"].append(str(key))

            num_video_rows = len(video_rows["imgname"])
            if audio_source == "frame" and frame_samples_per_video is not None and num_video_rows > frame_samples_per_video:
                keep = sorted(rng.choice(num_video_rows, size=frame_samples_per_video, replace=False).tolist())
            else:
                keep = list(range(num_video_rows))
            for row_index in keep:
                for field in rows:
                    rows[field].append(video_rows[field][row_index])

    if not rows["imgname"]:
        raise RuntimeError(f"No usable WSV scene-audio samples were extracted for {out_path}")

    long_audio = audio_source == "full"
    audio_codes = None
    if long_audio:
        first_by_video: dict[str, int] = {}
        for index, wsv_path in enumerate(rows["wsv_path"]):
            first_by_video.setdefault(str(wsv_path), index)
        unique_indices = list(first_by_video.values())
        print(
            f"Encoding {len(unique_indices)} unique full audio clips for {len(rows['audio_waveform'])} frame samples with {audio_encoder.encoder}",
            flush=True,
        )
        unique_feat = audio_encoder.encode_batch(
            [rows["audio_waveform"][index] for index in unique_indices],
            [rows["audio_sample_rate"][index] for index in unique_indices],
            full_audio=True,
        )
        feat_by_video = {str(rows["wsv_path"][index]): unique_feat[pos] for pos, index in enumerate(unique_indices)}
        audio_feat = np.stack([feat_by_video[str(wsv_path)] for wsv_path in rows["wsv_path"]], axis=0).astype(np.float32)
        if audio_encoder.encoder == "musicgen":
            code_shape = audio_encoder.code_shape
            if code_shape is None:
                raise RuntimeError("MusicGen audio encoder did not report an audio code shape.")
            audio_codes = audio_feat.reshape(audio_feat.shape[0], code_shape[0], code_shape[1]).astype(np.int32)
    else:
        print(f"Encoding {len(rows['audio_waveform'])} frame audio clips with {audio_encoder.encoder}", flush=True)
        audio_feat = audio_encoder.encode_batch(rows["audio_waveform"], rows["audio_sample_rate"], full_audio=False)
        if audio_encoder.encoder == "musicgen":
            code_shape = audio_encoder.code_shape
            if code_shape is None:
                raise RuntimeError("MusicGen audio encoder did not report an audio code shape.")
            audio_codes = audio_feat.reshape(audio_feat.shape[0], code_shape[0], code_shape[1]).astype(np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        imgname=np.asarray(rows["imgname"]),
        audio_feat=audio_feat.astype(np.float32),
        audio_rms_dbfs=np.asarray(rows["audio_rms_dbfs"], dtype=np.float32),
        bbox=np.asarray(rows["bbox"], dtype=np.float32),
        mask=np.stack([resize_binary_mask(mask) for mask in rows["mask"]], axis=0).astype(np.uint8),
        category=np.asarray(rows["category"], dtype=np.int64),
        wsv_path=np.asarray(rows["wsv_path"]),
        frame_key=np.asarray(rows["frame_key"]),
        audio_dim=np.asarray(audio_feat.shape[1], dtype=np.int32),
        audio_encoder=np.asarray(audio_encoder.encoder),
        audio_model_name=np.asarray(audio_encoder.model_name),
        audio_source=np.asarray("full_wav_audio" if audio_source == "full" else "scene_audio_channels_175_184"),
        semantic_source=np.asarray("channel_195"),
        audio_code_shape=np.asarray(audio_encoder.code_shape or (0, 0), dtype=np.int32),
        audio_codebook_size=np.asarray(audio_encoder.codebook_size, dtype=np.int32),
        audio_codec_sample_rate=np.asarray(audio_encoder.codec_sample_rate, dtype=np.int32),
        audio_codec_bandwidth=np.asarray(audio_encoder.codec_bandwidth, dtype=np.float32),
        audio_codec_seconds=np.asarray(audio_encoder.codec_seconds, dtype=np.float32),
    )
    if audio_codes is not None:
        payload["audio_codes"] = audio_codes.astype(np.int32)
    np.savez_compressed(out_path, **payload)
    if skipped_quiet:
        print(f"[audio-filter] skipped {skipped_quiet} quiet {audio_source} samples below {min_audio_rms_dbfs:.2f} dBFS", flush=True)
    print(f"Saved {len(rows['imgname'])} {audio_source} audio sparse samples -> {out_path}", flush=True)


def merge_npz_files(input_paths: list[Path], out_path: Path) -> None:
    if not input_paths:
        raise ValueError("No --merge-input files were provided.")
    sample_keys = ["imgname", "audio_feat", "audio_rms_dbfs", "bbox", "mask", "category", "wsv_path", "frame_key"]
    metadata_keys = [
        "audio_dim",
        "audio_encoder",
        "audio_model_name",
        "audio_source",
        "semantic_source",
        "audio_code_shape",
        "audio_codebook_size",
        "audio_codec_sample_rate",
        "audio_codec_bandwidth",
        "audio_codec_seconds",
    ]
    merged: dict[str, np.ndarray] = {}
    metadata: dict[str, np.ndarray] = {}
    for index, path in enumerate(input_paths):
        with np.load(path, allow_pickle=False) as data:
            if index == 0:
                for key in metadata_keys:
                    if key in data:
                        metadata[key] = np.asarray(data[key])
            for key in sample_keys:
                value = np.asarray(data[key])
                if key not in merged:
                    merged[key] = value
                else:
                    merged[key] = np.concatenate([merged[key], value], axis=0)
    if "audio_feat" in merged:
        metadata["audio_dim"] = np.asarray(merged["audio_feat"].shape[1], dtype=np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged, **metadata)
    print(f"Merged {len(input_paths)} shards, {len(merged['imgname'])} samples -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", action="append", default=[])
    parser.add_argument("--val-root", action="append", default=[])
    parser.add_argument("--out-dir", default="data/datasets")
    parser.add_argument("--train-name", default="wsv_audio_train.npz")
    parser.add_argument("--val-name", default="wsv_audio_val.npz")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--split-val-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--audio-dim", type=int, default=1024, help="Only used when --audio-encoder raw.")
    parser.add_argument("--audio-encoder", choices=["raw", "ast", "encodec", "musicgen"], default="ast")
    parser.add_argument("--audio-model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--audio-codec-seconds", type=float, default=10.0)
    parser.add_argument("--audio-codec-bandwidth", type=float, default=6.0)
    parser.add_argument("--audio-encoder-device", default="cuda")
    parser.add_argument("--audio-encoder-batch-size", type=int, default=16)
    parser.add_argument("--audio-source", choices=["frame", "full"], default="frame")
    parser.add_argument("--frame-samples-per-video", type=int, default=None)
    parser.add_argument("--min-audio-rms-dbfs", type=float, default=None, help="Drop audio samples quieter than this RMS dBFS threshold, e.g. -30. Use none to disable.")
    parser.add_argument("--frame-sample-seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-input", action="append", default=[])
    parser.add_argument("--merge-output", default=None)
    args = parser.parse_args()

    if args.merge_input:
        if args.merge_output is None:
            raise ValueError("--merge-output is required when --merge-input is used.")
        merge_npz_files([Path(path) for path in args.merge_input], Path(args.merge_output))
        return

    train_paths = iter_wsv_files(args.train_root)
    val_paths = iter_wsv_files(args.val_root)
    if not val_paths and train_paths:
        train_paths, val_paths = split_train_val(train_paths, args.split_val_fraction, args.split_seed)
    if args.max_videos is not None:
        train_paths = train_paths[: args.max_videos]
        val_paths = val_paths[: args.max_videos]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        train_paths = train_paths[args.shard_index :: args.num_shards]
        val_paths = val_paths[args.shard_index :: args.num_shards]
        print(
            f"[shard] index={args.shard_index}/{args.num_shards} "
            f"train_videos={len(train_paths)} val_videos={len(val_paths)}",
            flush=True,
        )

    device = args.audio_encoder_device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
    audio_encoder = AudioFeatureEncoder(
        args.audio_encoder,
        args.audio_dim,
        args.audio_model_name,
        device,
        args.audio_encoder_batch_size,
        args.audio_codec_seconds,
        args.audio_codec_bandwidth,
    )

    out_dir = Path(args.out_dir)
    build_dataset(
        train_paths,
        out_dir / args.train_name,
        args.frame_stride,
        args.max_frames_per_video,
        audio_encoder,
        args.audio_source,
        args.frame_samples_per_video,
        args.frame_sample_seed,
        args.min_audio_rms_dbfs,
    )
    if val_paths:
        build_dataset(
            val_paths,
            out_dir / args.val_name,
            args.frame_stride,
            args.max_frames_per_video,
            audio_encoder,
            args.audio_source,
            args.frame_samples_per_video,
            args.frame_sample_seed + 1,
            args.min_audio_rms_dbfs,
        )


if __name__ == "__main__":
    main()
