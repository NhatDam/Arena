"""Inference helper for LeLaN controllers."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml

import clip

from arena_ai_integration.models.lelan.lelan import DenseNetwork_lelan, LeLaN_clip
from arena_ai_integration.models.lelan.lelan_comp import LeLaN_clip_FiLM, replace_bn_with_gn


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
            if hasattr(value, 'state_dict'):
                checkpoint = value.state_dict()
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError('LeLaN checkpoint does not contain a state_dict-compatible object.')

    prefixes = ('module.', 'model.')
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                    break
        normalized[new_key] = value
    return normalized


class LeLaNInferenceModel:
    """Runtime wrapper around a LeLaN checkpoint."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda',
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.config = load_config(config_path)

        model_type = str(self.config.get('model_type', 'lelan'))
        if model_type != 'lelan':
            raise ValueError(
                "The Arena LeLaN DWB integration currently supports model_type='lelan' only. "
                f"Got: {model_type}"
            )
        if self.config.get('vision_encoder') != 'lelan_clip_film':
            raise ValueError(f"Unsupported LeLaN vision_encoder: {self.config.get('vision_encoder')}")

        vision_encoder = LeLaN_clip_FiLM(
            obs_encoding_size=self.config['encoding_size'],
            context_size=self.config['context_size'],
            mha_num_attention_heads=self.config['mha_num_attention_heads'],
            mha_num_attention_layers=self.config['mha_num_attention_layers'],
            mha_ff_dim_factor=self.config['mha_ff_dim_factor'],
            feature_size=self.config['feature_size'],
            clip_type=self.config['clip_type'],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        text_encoder, _ = clip.load(self.config['clip_type'], device=self.device)
        dist_pred_network = DenseNetwork_lelan(
            embedding_dim=self.config['encoding_size'],
            control_horizon=self.config['len_traj_pred'],
        )
        self.model = LeLaN_clip(
            vision_encoder=vision_encoder,
            dist_pred_net=dist_pred_network,
            text_encoder=text_encoder.float(),
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = _extract_state_dict(checkpoint)
        self.load_report = self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.eval_text_encoder()

        self.model_type = model_type
        self.context_size = int(self.config['context_size'])
        self.len_traj_pred = int(self.config['len_traj_pred'])
        self.image_size = tuple(int(v) for v in self.config['image_size'])
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def _preprocess_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}")

        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image_rgb))
            .to(self.device)
            .float()
            .permute(2, 0, 1)
            / 255.0
        )
        image_tensor = TF.resize(image_tensor, list(self.image_size[::-1]), antialias=True)
        return (image_tensor - self.mean) / self.std

    @torch.no_grad()
    def predict(
        self,
        image_history: list[np.ndarray],
        instruction: str,
        waypoint_scale: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        if len(image_history) < self.context_size:
            raise ValueError(f"Expected {self.context_size} images, got {len(image_history)}")
        current_img = self._preprocess_image(image_history[-1]).unsqueeze(0)
        tokens = clip.tokenize([instruction], truncate=True).to(self.device)
        feat_text = self.model('text_encoder', inst_ref=tokens)

        obsgoal_cond = self.model(
            'vision_encoder',
            obs_img=current_img,
            feat_text=feat_text.to(dtype=torch.float32),
        )

        waypoints = self.model('dist_pred_net', obsgoal_cond=obsgoal_cond)[0]
        if waypoint_scale != 1.0:
            waypoints = waypoints * float(waypoint_scale)

        return waypoints.detach().cpu().numpy(), 0.0
