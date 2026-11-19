from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce

from score_hmr.diffusion import GaussianDiffusion
from score_hmr.models import FC
from score_hmr.utils.geometry import aa_to_rotmat
from score_hmr.utils.model_blocks import ResMLPBlock, SinusoidalPosEmb
from score_hmr.utils.utils import (
    default,
    extract,
    identity,
    normalize_betas,
)


class FCWithBBox(FC):
    """ScoreHMR denoiser with an extra bbox branch.

    The SMPL pose/beta branches remain the official FC model. The extra four
    dimensions predict normalized bbox parameters: cx, cy, w, h in image space.
    This keeps training one-sample-per-person while letting the model learn
    where that person is in the original frame.
    """

    bbox_dim = 4

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.base_diffusion_dim = self.diffusion_dim
        self.diffusion_dim = self.base_diffusion_dim + self.bbox_dim

        hidden_dim = cfg.MODEL.DENOISING_MODEL.HIDDEN_LAYER_DIM
        img_feats = cfg.MODEL.DENOISING_MODEL.IMG_FEATS
        cond_dim = self.betas_emb_dim if self.split_img_emb and img_feats == "pare" else self.thetas_emb_dim
        time_dim = self.bbox_dim * 4
        self.time_mlp_bbox = nn.Sequential(
            SinusoidalPosEmb(self.bbox_dim),
            nn.Linear(self.bbox_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_mlp_bbox = nn.Linear(self.bbox_dim, self.bbox_dim)
        self.blocks_bbox = nn.ModuleList(
            [
                ResMLPBlock(
                    input_dim=self.bbox_dim,
                    hidden_dim=hidden_dim,
                    time_emb_dim=time_dim,
                    cond_emb_dim=cond_dim,
                )
                for _ in range(max(1, cfg.MODEL.DENOISING_MODEL.NUM_BLOCKS_SHAPE))
            ]
        )
        self.final_mlp_bbox = nn.Linear(self.bbox_dim, self.bbox_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond_emb: torch.Tensor | None = None) -> torch.Tensor:
        base_x = x[:, : self.base_diffusion_dim]
        bbox = x[:, self.base_diffusion_dim :]
        base_out = super().forward(base_x, time, cond_emb)

        bbox_emb = cond_emb
        if self.split_img_emb and cond_emb is not None:
            bbox_emb = cond_emb[:, 3072:]
        bbox = self.init_mlp_bbox(bbox)
        tt_bbox = self.time_mlp_bbox(time)
        for block in self.blocks_bbox:
            bbox = block(bbox, tt_bbox, bbox_emb)
        bbox = self.final_mlp_bbox(bbox)
        return torch.cat((base_out, bbox), dim=1)


class GaussianDiffusionWithBBox(GaussianDiffusion):
    bbox_dim = 4

    def __init__(self, cfg, model: nn.Module, **kwargs) -> None:
        super().__init__(cfg, model, **kwargs)
        self.use_bbox = True

    def _build_clean_params(self, batch: dict) -> torch.Tensor:
        batch_size = batch["img_feats"].size(0)
        global_orient_aa = batch["smpl_params"]["global_orient"]
        body_pose_aa = batch["smpl_params"]["body_pose"]

        global_orient_6d = (
            aa_to_rotmat(global_orient_aa.reshape(-1, 3))
            .reshape(batch_size, -1, 3, 3)[:, :, :, :2]
            .permute(0, 1, 3, 2)
            .reshape(batch_size, 1, -1)
        )
        body_pose_6d = (
            aa_to_rotmat(body_pose_aa.reshape(-1, 3))
            .reshape(batch_size, -1, 3, 3)[:, :, :, :2]
            .permute(0, 1, 3, 2)
            .reshape(batch_size, 23, -1)
        )
        pose = torch.cat((global_orient_6d, body_pose_6d), dim=1).reshape(batch_size, -1)

        if self.use_betas:
            scaled_betas = normalize_betas(batch["smpl_params"]["betas"], self.betas_min, self.betas_max)
            params = torch.cat((pose, scaled_betas), dim=1)
        else:
            params = pose

        bbox_params = batch["bbox_params"].to(params.dtype).clamp(0.0, 1.0) * 2.0 - 1.0
        return torch.cat((params, bbox_params), dim=1)

    def p_losses(self, batch: dict, t_: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self._build_clean_params(batch)
        noise = default(noise, lambda: torch.randn_like(params))
        cond_feats = batch["img_feats"]

        noised_params = self.q_sample(x_start=params, t_=t_, noise=noise)
        pred_noise, pred_x_start = self.model_predictions(noised_params, t_, cond_feats)

        if self.objective == "pred_noise":
            target = noise
            model_output = pred_noise
        elif self.objective == "pred_x0":
            target = params
            model_output = pred_x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        simple_loss = self.loss_fn(model_output, target, reduction="none")
        simple_loss = reduce(simple_loss, "b ... -> b (...)", "mean")
        simple_loss = simple_loss * extract(self.p2_loss_weight, t_, simple_loss.shape)
        return simple_loss, pred_x_start

    def model_predictions(
        self,
        x: torch.Tensor,
        t_: torch.Tensor,
        cond_feats: torch.Tensor,
        batch: dict = None,
        clip_x_start: bool = True,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        model_output = self.model(x, t_, cond_feats)

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t_, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t_, x_start)
        else:
            raise ValueError(f"unknown objective {self.objective}")
        return pred_noise, x_start
