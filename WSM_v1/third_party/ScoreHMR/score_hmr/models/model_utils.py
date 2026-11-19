import torch
from pathlib import Path
from collections import OrderedDict
from ema_pytorch import EMA
from . import SMPL, FC
from .pare import PARE
from ..diffusion import GaussianDiffusion
from constants import CHECKPOINT_DIR, RESULTS_DIR, PARE_CHECKPOINT


def load_diffusion_model(cfg, **kwargs):
    name = kwargs.get("name", "score_hmr")
    milestone = kwargs.get("milestone", 100)
    use_default_ckpt = kwargs.get('use_default_ckpt', False)
    device = kwargs.get("device", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Set up diffusion model.
    model = FC(cfg).to(device)
    diffusion_model = GaussianDiffusion(cfg, model, **kwargs).to(device)

    # Load model weights.
    ckpt_dir = f"{CHECKPOINT_DIR}/{name}" if use_default_ckpt else f"{RESULTS_DIR}/checkpoints/{name}"
    candidates = (
        Path(ckpt_dir) / f"model-{milestone}.pt",
        Path(ckpt_dir) / f"model-step_{milestone:07d}.pt",
        Path(ckpt_dir) / f"model-step_{milestone * int(cfg.TRAIN.LOG_FREQ):07d}.pt",
    )
    checkpoint = next((path for path in candidates if path.is_file()), candidates[0])
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    ema = EMA(diffusion_model, beta=0.995, update_every=10)
    ema.load_state_dict(data["ema"])
    ema_model = ema.ema_model
    # Set up SMPL model.
    smpl_cfg = {k.lower(): v for k, v in dict(cfg.SMPL).items()}
    ema_model.smpl = SMPL(**smpl_cfg).to(device)
    ema_model.eval()

    return ema_model


def load_pare(cfg_SMPL):
    # Create model.
    pare = PARE(
        backbone="hrnet_w32-conv",
        shape_input_type="feats.shape.cam",
        pose_input_type="feats.self_pose.shape.cam",
        use_heatmaps="part_segm",
        use_keypoint_attention=True,
        num_deconv_layers=2,
        num_deconv_filters=128,
    )

    # Load weights from the given checkpoint.
    state_dict = torch.load(PARE_CHECKPOINT, weights_only=False)["state_dict"]
    pretrained_keys = state_dict.keys()
    new_state_dict = OrderedDict()
    for pk in pretrained_keys:
        if pk.startswith("model."):
            new_state_dict[pk.replace("model.", "")] = state_dict[pk]
        else:
            new_state_dict[pk] = state_dict[pk]
    pare.load_state_dict(new_state_dict, strict=False)

    smpl_cfg = {k.lower(): v for k, v in dict(cfg_SMPL).items()}
    pare.smpl = SMPL(**smpl_cfg)

    return pare
