import torch
import torch.nn as nn
from arena_ai_integration.models.socialnav.urban_film import (
    FiLMNetwork,
    PolarEmbedding,
    PositionalEncoding,
    replace_bn_with_gn,
)


class PedestrianEncoder(nn.Module):
    """
    Encode pedestrians positions over T frames (no ID) -> human_tokens (B, T*P, D)
    """
    def __init__(self, attn_dim: int, num_freqs: int, T_human: int,
                 num_layers: int = 2, num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_dim = attn_dim
        self.T_human = T_human
        self.polar = PolarEmbedding(num_freqs)
        self.coord_proj = nn.Linear(self.polar.out_dim, attn_dim)
        self.time_emb = nn.Embedding(T_human, attn_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dim_feedforward=attn_dim * ff_mult,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, human_pos: torch.Tensor, human_mask: torch.Tensor | None = None):
        """
        human_pos: (B, T, P, 2)
        human_mask: (B, T, P) with True = padded/invalid (optional)
        return:
            human_tokens: (B, T*P, D)
            key_padding_mask: (B, T*P) with True = ignore
            has_human: (B,) bool — True if sample has at least one valid pedestrian
        """
        B, T, P, _ = human_pos.shape
        assert T == self.T_human, f"Expected T_human={self.T_human}, got T={T}"

        # Build key_padding_mask
        if human_mask is not None:
            assert human_mask.shape == (B, T, P)
            kpm = human_mask.reshape(B, T * P)  # True = ignore
        else:
            kpm = torch.zeros(B, T * P, dtype=torch.bool, device=human_pos.device)

        # Determine which samples have at least one valid human
        has_human = ~kpm.all(dim=1)  # (B,) True = has at least one valid token

        # ---------- Early return if NO sample has humans ----------
        if not has_human.any():
            # Return zeros — no computation needed, no NaN risk
            human_tokens = torch.zeros(B, T * P, self.attn_dim,
                                       device=human_pos.device, dtype=human_pos.dtype)
            return human_tokens, kpm, has_human

        # ---------- Process only samples that have humans ----------
        # Polar embed per position
        x = human_pos.reshape(B, T * P, 2)
        x = self.polar(x)
        x = self.coord_proj(x)

        # Add time embedding
        t_ids = torch.arange(T, device=human_pos.device).view(1, T, 1).expand(B, T, P)
        t_emb = self.time_emb(t_ids).reshape(B, T * P, self.attn_dim)
        x = x + t_emb

        # For samples with ALL tokens masked, we must avoid feeding them
        # into the transformer (softmax over all -inf = NaN).
        # Strategy: process only valid samples through the encoder.
        if has_human.all():
            # All samples have humans — process normally
            x = self.encoder(x, src_key_padding_mask=kpm)
        else:
            # Mixed batch: only encode samples that have valid humans
            valid_idx = has_human.nonzero(as_tuple=True)[0]  # indices of valid samples
            x_valid = x[valid_idx]                           # (V, TP, D)
            kpm_valid = kpm[valid_idx]                       # (V, TP)

            x_valid = self.encoder(x_valid, src_key_padding_mask=kpm_valid)

            # Write back; invalid samples stay as their pre-encoder values (won't be used)
            # But safer to zero them out
            out = torch.zeros_like(x)
            out[valid_idx] = x_valid
            x = out

        return x, kpm, has_human


class CrossAttentionBlock(nn.Module):
    def __init__(self, attn_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(attn_dim)
        self.ln_kv = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(
            attn_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.ln_ff = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, 4 * attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * attn_dim, attn_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens, kv_tokens, kv_key_padding_mask=None):
        """
        q_tokens: (B, S, D)
        kv_tokens: (B, H, D)
        kv_key_padding_mask: (B, H) True=ignore
        NOTE: caller must ensure not ALL kv tokens are masked for any sample
              in the batch, OR caller should handle it externally.
        """
        q = self.ln_q(q_tokens)
        kv = self.ln_kv(kv_tokens)
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=kv_key_padding_mask
        )
        delta = self.drop(attn_out)
        x = q_tokens + delta
        x = x + self.ff(self.ln_ff(x))
        return x


class SocialUrbanNavFiLM(nn.Module):
    """
    Main tokens (FiLM visual+text + ego history tokens + text token)
      -> cross-attend to human tokens
      -> transformer encoder
      -> output K trajectories and K arrived logits
    """
    def __init__(self,
                 context_size: int,
                 len_traj_pred: int,
                 num_freqs: int,
                 attn_dim: int,
                 num_attn_layers: int,
                 num_attn_heads: int,
                 ff_dim_factor: int,
                 dropout: float,
                 K: int = 5,
                 T_human: int = 8,
                 human_num_layers: int = 2,
                 clip_type: str = "ViT-B/32",
                 ):
        super().__init__()
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.attn_dim = attn_dim
        self.K = K
        self.T_human = T_human

        # --- FiLM backbone ---
        if clip_type == "ViT-B/32":
            self.obsgoal_encoder = FiLMNetwork(8, 128, 512)
            text_in_dim = 512
        elif clip_type == "ViT-L/14@336px":
            self.obsgoal_encoder = FiLMNetwork(8, 128, 768)
            text_in_dim = 512
        elif clip_type == "RN50x64":
            self.obsgoal_encoder = FiLMNetwork(8, 128, 1024)
            text_in_dim = 512
        else:
            raise ValueError(f"Unknown clip_type: {clip_type}")

        self.obsgoal_encoder = replace_bn_with_gn(self.obsgoal_encoder)
        # self.obsgoal_compress = nn.LazyLinear(attn_dim)
        self.obsgoal_compress = nn.Linear(4096, attn_dim)
        self.text_compress = nn.Sequential(
            nn.Linear(text_in_dim, attn_dim),
            nn.BatchNorm1d(attn_dim)
        )

        # ego history encoder
        self.ego_polar = PolarEmbedding(num_freqs)
        self.ego_proj = nn.Linear(self.ego_polar.out_dim, attn_dim)

        # Feature fusion MLP
        self.linear1 = nn.Linear(attn_dim, ff_dim_factor * attn_dim)
        self.bn1 = nn.BatchNorm1d(ff_dim_factor * attn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim_factor * attn_dim, attn_dim)
        self.bn2 = nn.BatchNorm1d(attn_dim)

        # Main positional encoding
        self.main_seq_len = context_size + 1
        self.positional_encoding = PositionalEncoding(attn_dim, self.main_seq_len)

        # --- Human branch ---
        self.ped_encoder = PedestrianEncoder(
            attn_dim=attn_dim,
            num_freqs=num_freqs,
            T_human=T_human,
            num_layers=human_num_layers,
            num_heads=max(1, num_attn_heads // 2),
            ff_mult=ff_dim_factor,
            dropout=dropout
        )

        # --- Cross attention ---
        self.cross_block = CrossAttentionBlock(
            attn_dim, num_heads=num_attn_heads, dropout=dropout
        )

        # --- Main transformer encoder ---
        sa_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_attn_heads,
            dim_feedforward=attn_dim * ff_dim_factor,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout
        )
        self.sa_encoder = nn.TransformerEncoder(sa_layer, num_layers=num_attn_layers)

        # --- Decoder + heads ---
        self.mlp_decoder = nn.Sequential(
            nn.Linear(self.main_seq_len * attn_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.wp_head = nn.Linear(64, K * len_traj_pred * 2)
        self.arrived_head = nn.Linear(64, K)

    def _token_fusion_mlp(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.linear1(tokens).permute(0, 2, 1)
        x = self.relu(self.bn1(x).permute(0, 2, 1))
        x = self.linear2(x).permute(0, 2, 1)
        x = self.bn2(x).permute(0, 2, 1)
        return x

    def forward(self,
                text_feat: torch.Tensor,
                ego_hist_xy: torch.Tensor,
                human_pos: torch.Tensor,
                curr_obs_img: torch.Tensor,
                human_mask: torch.Tensor | None = None,
                ):
        B = text_feat.size(0)

        # --- Visual token ---
        obsgoal_fmap = self.obsgoal_encoder(curr_obs_img, text_feat)
        obsgoal_token = self.obsgoal_compress(
            obsgoal_fmap.flatten(start_dim=1)
        ).unsqueeze(1)

        # --- Text token ---
        text_token = self.text_compress(text_feat).unsqueeze(1)

        # --- Ego history tokens ---
        ego_tok = self.ego_proj(self.ego_polar(ego_hist_xy))
        ego_tok = ego_tok[:, :-1, :]

        # --- Main tokens ---
        main_tokens = torch.cat([obsgoal_token, ego_tok, text_token], dim=1)
        main_tokens = self._token_fusion_mlp(main_tokens)
        main_tokens = self.positional_encoding(main_tokens)

        # ============================================================
        # HUMAN BRANCH — NaN-safe
        # ============================================================
        human_tokens, human_kpm, has_human = self.ped_encoder(human_pos, human_mask)

        if has_human.any():
            # At least one sample in the batch has humans
            if has_human.all():
                # ALL samples have humans — straightforward cross attention
                attended = self.cross_block(
                    main_tokens, human_tokens,
                    kv_key_padding_mask=human_kpm
                )
                main_tokens = attended
            else:
                # MIXED batch — only cross-attend for samples with humans
                valid_idx = has_human.nonzero(as_tuple=True)[0]
                invalid_idx = (~has_human).nonzero(as_tuple=True)[0]

                # Cross-attend only valid samples
                q_valid = main_tokens[valid_idx]          # (V, S, D)
                kv_valid = human_tokens[valid_idx]        # (V, TP, D)
                kpm_valid = human_kpm[valid_idx]          # (V, TP)

                attended_valid = self.cross_block(
                    q_valid, kv_valid,
                    kv_key_padding_mask=kpm_valid
                )

                # Reassemble: valid samples get cross-attended tokens,
                #              invalid samples keep original main_tokens
                main_tokens_out = main_tokens.clone()
                main_tokens_out[valid_idx] = attended_valid
                # main_tokens_out[invalid_idx] already has original main_tokens
                main_tokens = main_tokens_out
        # else: no humans at all — main_tokens pass through unchanged

        # --- Main transformer ---
        feat = self.sa_encoder(main_tokens)

        # --- Decode ---
        h = self.mlp_decoder(feat.reshape(B, -1))
        wp = self.wp_head(h).reshape(B, self.K, self.len_traj_pred, 2)
        arrived_logits = self.arrived_head(h).reshape(B, self.K)

        return wp, arrived_logits, feat
