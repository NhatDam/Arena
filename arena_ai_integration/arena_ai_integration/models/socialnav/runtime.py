"""Inference wrappers for SocialNav and UrbanNav."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import clip

from arena_ai_integration.models.socialnav.social_film import SocialUrbanNavFiLM
from arena_ai_integration.models.socialnav.urban_ca import UrbanNavCrossAttention
from arena_ai_integration.models.socialnav.urban_film import UrbanNavFiLM
from arena_ai_integration.models.socialnav.urban_mlp import UrbanNavMLP


if not hasattr(F, 'scaled_dot_product_attention'):
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        scale = query.size(-1) ** -0.5
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, value)

    F.scaled_dot_product_attention = scaled_dot_product_attention


URBANNAV_MODELS = {
    'mlp': UrbanNavMLP,
    'cross_attention': UrbanNavCrossAttention,
    'film': UrbanNavFiLM,
    'social_film': SocialUrbanNavFiLM,
}


def _ensure_rgb_uint8_image(image) -> np.ndarray:
    """Return a contiguous HxWx3 uint8 numpy image for OpenCV/Torch."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Expected image convertible to HxWx3 array, got type={type(image).__name__} "
            f"shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}"
        )

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)


def _load_state_dict(checkpoint: str) -> dict:
    try:
        return torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=torch.device('cpu'))


def setup_model(
    model_type: str,
    checkpoint: str,
    context_size: int,
    len_traj_pred: int,
    visual_feat_size: int,
    num_freqs: int,
    attn_dim: int,
    num_attn_layers: int,
    num_attn_heads: int,
    ff_dim_factor: int,
    dropout: float,
    film_feature_size: Optional[int] = None,
    clip_type: Optional[str] = None,
    trajectory_modes: int = 5,
) -> nn.Module:
    if model_type not in URBANNAV_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == 'social_film':
        model = SocialUrbanNavFiLM(
            context_size=context_size,
            len_traj_pred=len_traj_pred,
            num_freqs=num_freqs,
            attn_dim=attn_dim,
            num_attn_layers=num_attn_layers,
            num_attn_heads=num_attn_heads,
            ff_dim_factor=ff_dim_factor,
            dropout=dropout,
            K=trajectory_modes,
            T_human=8,
            human_num_layers=2,
            clip_type=clip_type,
        )
    else:
        model = URBANNAV_MODELS[model_type](
            context_size=context_size,
            len_traj_pred=len_traj_pred,
            visual_feat_size=visual_feat_size,
            num_freqs=num_freqs,
            attn_dim=attn_dim,
            num_attn_layers=num_attn_layers,
            num_attn_heads=num_attn_heads,
            ff_dim_factor=ff_dim_factor,
            dropout=dropout,
            film_feature_size=film_feature_size,
            clip_type=clip_type,
        )

    if checkpoint:
        model.load_state_dict(_load_state_dict(checkpoint))
        print(f"Loaded checkpoint from {checkpoint}")

    return model


class UrbanNavModel:
    """Wrapper for UrbanNav model with proper initialization."""

    def __init__(self, config_path: str, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        self.model = setup_model(
            model_type=self.config['model']['feature_fusion'],
            checkpoint=model_path,
            context_size=self.config['context_size'],
            len_traj_pred=self.config['len_traj_pred'],
            visual_feat_size=self.config['model']['visual_feat_size'],
            num_freqs=self.config['model']['num_freqs'],
            attn_dim=self.config['model']['attn_dim'],
            num_attn_layers=self.config['model']['num_attn_layers'],
            num_attn_heads=self.config['model']['num_attn_heads'],
            ff_dim_factor=self.config['model']['ff_dim_factor'],
            dropout=self.config['model']['dropout'],
            film_feature_size=self.config['model'].get('film_feat_size'),
            clip_type=self.config['model'].get('clip_type'),
        ).to(self.device)
        self.model.eval()

        self.clip_model, self.clip_preprocess = clip.load(
            self.config['model']['clip_type'],
            device=self.device,
        )
        self.clip_model.eval()

        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dinov2_model = self.dinov2_model.to(self.device)
        self.dinov2_model.eval()

    def encode_instruction(self, instruction: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(instruction).to(self.device)
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        image = _ensure_rgb_uint8_image(image)
        h, w = image.shape[:2]
        new_h = max(14, (h // 14) * 14)
        new_w = max(14, (w // 14) * 14)
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h))

        img_tensor = torch.from_numpy(image).float().to(self.device)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            features = self.dinov2_model(img_tensor)
        return features

    def predict(
        self,
        image_history: List[np.ndarray],
        instruction: str,
        human_positions: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, float]:
        assert len(image_history) == self.config['context_size']

        with torch.no_grad():
            visual_feats = [self.encode_image(img) for img in image_history]
            visual_feats = torch.cat(visual_feats, dim=0)
            text_feat = self.encode_instruction(instruction)

            curr_img = cv2.resize(_ensure_rgb_uint8_image(image_history[-1]), (256, 256))
            curr_obs_img = torch.from_numpy(curr_img).float().to(self.device)
            curr_obs_img = curr_obs_img / 255.0
            curr_obs_img = curr_obs_img.permute(2, 0, 1).unsqueeze(0)

            if self.config['model']['feature_fusion'] == 'film':
                human_local = {}
                if human_positions is not None:
                    human_local = {
                        i: {'pos': torch.from_numpy(h).float().to(self.device)}
                        for i, h in enumerate(human_positions)
                    }
                output = self.model(
                    text_feat,
                    visual_feats.unsqueeze(0),
                    human_local=human_local,
                    curr_obs_img=curr_obs_img,
                )
            else:
                output = self.model(
                    text_feat,
                    visual_feats.unsqueeze(0),
                    curr_obs_img=curr_obs_img,
                )

            if isinstance(output, (tuple, list)):
                waypoints = output[0]
                arrival_logit = output[1][0]
            else:
                waypoints = output.get('waypoints', output[0])
                arrival_logit = output.get('arrival', torch.zeros(1, device=self.device))[0]

            arrival_score = torch.sigmoid(arrival_logit)
            waypoints = waypoints[0]

        return waypoints.cpu().numpy(), arrival_score.cpu().item()


class SocialNavModel:
    """Wrapper for SocialUrbanNav model with human-aware prediction."""

    def __init__(self, config_path: str, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        self.trajectory_modes = self._resolve_trajectory_modes(model_path)
        self.model = setup_model(
            model_type='social_film',
            checkpoint=model_path,
            context_size=self.config['context_size'],
            len_traj_pred=self.config['len_traj_pred'],
            visual_feat_size=self.config['model']['visual_feat_size'],
            num_freqs=self.config['model']['num_freqs'],
            attn_dim=self.config['model']['attn_dim'],
            num_attn_layers=self.config['model']['num_attn_layers'],
            num_attn_heads=self.config['model']['num_attn_heads'],
            ff_dim_factor=self.config['model']['ff_dim_factor'],
            dropout=self.config['model']['dropout'],
            film_feature_size=self.config['model'].get('film_feat_size'),
            clip_type=self.config['model'].get('clip_type'),
            trajectory_modes=self.trajectory_modes,
        ).to(self.device)
        self.model.eval()

        self.clip_model, self.clip_preprocess = clip.load(
            self.config['model']['clip_type'],
            device=self.device,
        )
        self.clip_model.eval()

        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dinov2_model = self.dinov2_model.to(self.device)
        self.dinov2_model.eval()

    def _resolve_trajectory_modes(self, model_path: str) -> int:
        checkpoint_name = Path(model_path).name
        if checkpoint_name == 'SocialNav_1_path.pth':
            return 1
        if checkpoint_name == 'SocialNav_margin_last.pth':
            return 5
        return int(self.config.get('model', {}).get('K', 5))

    def encode_instruction(self, instruction: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(instruction).to(self.device)
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        image = _ensure_rgb_uint8_image(image)
        h, w = image.shape[:2]
        new_h = max(14, (h // 14) * 14)
        new_w = max(14, (w // 14) * 14)
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h))

        img_tensor = torch.from_numpy(image).float().to(self.device)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            features = self.dinov2_model(img_tensor)
        return features

    def _forward(
        self,
        image_history: List[np.ndarray],
        instruction: str,
        human_positions: Optional[np.ndarray] = None,
        ego_hist_xy: Optional[np.ndarray] = None,
        human_mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(image_history) == self.config['context_size']

        with torch.no_grad():
            visual_feats = [self.encode_image(img) for img in image_history]
            visual_feats = torch.cat(visual_feats, dim=0)
            text_feat = self.encode_instruction(instruction)

            curr_img = cv2.resize(_ensure_rgb_uint8_image(image_history[-1]), (256, 256))
            curr_obs_img = torch.from_numpy(curr_img).float().to(self.device)
            curr_obs_img = curr_obs_img / 255.0
            curr_obs_img = curr_obs_img.permute(2, 0, 1).unsqueeze(0)

            if human_positions is not None:
                if isinstance(human_positions, np.ndarray):
                    human_pos_tensor = torch.from_numpy(human_positions).float().to(self.device)
                else:
                    human_pos_tensor = human_positions.to(self.device)

                if human_pos_tensor.ndim == 3:
                    human_pos_tensor = human_pos_tensor.unsqueeze(0)
                elif human_pos_tensor.ndim == 2:
                    human_pos_tensor = human_pos_tensor.unsqueeze(0).unsqueeze(0)

                if human_mask is not None:
                    if isinstance(human_mask, np.ndarray):
                        human_mask_tensor = torch.from_numpy(human_mask).bool().to(self.device)
                    else:
                        human_mask_tensor = human_mask.bool().to(self.device)

                    if human_mask_tensor.ndim == 2:
                        human_mask_tensor = human_mask_tensor.unsqueeze(0)
                    elif human_mask_tensor.ndim == 1:
                        human_mask_tensor = human_mask_tensor.unsqueeze(0).unsqueeze(0)
                else:
                    human_mask_tensor = torch.zeros(
                        human_pos_tensor.shape[:-1],
                        dtype=torch.bool,
                        device=self.device,
                    )
            else:
                batch_size = 1
                history_len = self.config['context_size']
                human_pos_tensor = torch.zeros(batch_size, history_len, 0, 2, dtype=torch.float32).to(self.device)
                human_mask_tensor = torch.ones(batch_size, history_len, 0, dtype=torch.bool).to(self.device)

            if self.config['model']['feature_fusion'] == 'social_film':
                if ego_hist_xy is not None:
                    ego_hist_xy_tensor = torch.from_numpy(ego_hist_xy).unsqueeze(0).to(self.device)
                else:
                    ego_hist_xy_tensor = torch.zeros(
                        1,
                        self.config['context_size'],
                        2,
                        dtype=torch.float32,
                    ).to(self.device)

                output = self.model(
                    text_feat=text_feat,
                    ego_hist_xy=ego_hist_xy_tensor,
                    human_pos=human_pos_tensor,
                    curr_obs_img=curr_obs_img,
                    human_mask=human_mask_tensor,
                )
            else:
                output = self.model(
                    text_feat=text_feat,
                    curr_obs_img=curr_obs_img,
                )

            if isinstance(output, (tuple, list)):
                waypoints, arrival_logit = output[0], output[1]
            else:
                waypoints = output
                arrival_logit = torch.zeros(1, dtype=torch.float32).to(self.device)

            if waypoints.ndim == 3:
                waypoints = waypoints.unsqueeze(1)
            if arrival_logit.ndim == 1:
                arrival_logit = arrival_logit.unsqueeze(0)

        return waypoints[0], torch.sigmoid(arrival_logit[0])

    def predict_candidates(
        self,
        image_history: List[np.ndarray],
        instruction: str,
        human_positions: Optional[np.ndarray] = None,
        ego_hist_xy: Optional[np.ndarray] = None,
        human_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        waypoints, arrival_scores = self._forward(
            image_history,
            instruction,
            human_positions=human_positions,
            ego_hist_xy=ego_hist_xy,
            human_mask=human_mask,
        )
        return waypoints.cpu().numpy(), arrival_scores.cpu().numpy()

    def predict(
        self,
        image_history: List[np.ndarray],
        instruction: str,
        human_positions: Optional[np.ndarray] = None,
        ego_hist_xy: Optional[np.ndarray] = None,
        human_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        candidates, arrival_scores = self.predict_candidates(
            image_history,
            instruction,
            human_positions=human_positions,
            ego_hist_xy=ego_hist_xy,
            human_mask=human_mask,
        )
        return candidates[0], float(arrival_scores[0])
