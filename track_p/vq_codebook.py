"""VQ-VAE codebook with EMA update and commitment loss.

See spec §7 (training) and van den Oord et al. 2017. EMA path avoids dead
codes under dense-signal training.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class VQCodebook(nn.Module):
    """Codebook of `size` embeddings of `dim` dimensions.

    - quantize(z): returns (indices, quantized_z, commitment_loss).
    - straight-through gradient: grad flows from quantized to z unchanged.
    - ema=True: embeddings are updated by EMA of assigned vectors (no gradient).
    - ema=False: embeddings are a regular nn.Parameter (baseline).
    """

    embeddings: Tensor | nn.Parameter
    ema_cluster_size: Tensor
    ema_embed_sum: Tensor
    usage_counter: Tensor

    def __init__(
        self,
        size: int,
        dim:  int,
        *,
        commitment_beta: float = 0.25,
        ema:             bool  = True,
        decay:           float = 0.99,
    ) -> None:
        super().__init__()
        self.size            = size
        self.dim             = dim
        self.commitment_beta = commitment_beta
        self.ema             = ema
        self.decay           = decay

        init = torch.randn(size, dim) * 0.1

        if ema:
            self.register_buffer("embeddings", init)
            self.register_buffer("ema_cluster_size", torch.zeros(size))
            self.register_buffer("ema_embed_sum",    init.clone())
        else:
            self.embeddings = nn.Parameter(init)  # type: ignore[assignment]

        self.register_buffer("usage_counter", torch.zeros(size, dtype=torch.long))

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # z: [B, dim]. Compute squared distance to every codebook vector.
        dist = torch.cdist(z, self.embeddings)               # [B, size]
        indices   = dist.argmin(dim=-1)                      # [B]
        quantized = self.embeddings[indices]                 # [B, dim]

        # Usage tracking
        for i in indices.tolist():
            self.usage_counter[i] += 1

        # Commitment loss (Oord 2017 eq. 3).
        commit_loss = self.commitment_beta * ((z - quantized.detach()) ** 2).mean()
        codebook_loss = ((quantized - z.detach()) ** 2).mean()
        loss = commit_loss + codebook_loss

        # EMA update of embeddings (no gradient path).
        if self.ema and self.training:
            with torch.no_grad():
                onehot = torch.zeros(z.shape[0], self.size, device=z.device)
                onehot.scatter_(1, indices.unsqueeze(1), 1)

                self.ema_cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay
                )
                self.ema_embed_sum.mul_(self.decay).add_(
                    onehot.T @ z, alpha=1 - self.decay
                )
                n = self.ema_cluster_size.sum()
                cluster = (self.ema_cluster_size + 1e-5) / (n + self.size * 1e-5) * n
                self.embeddings = self.ema_embed_sum / cluster.unsqueeze(1)

        # Straight-through estimator.
        quantized = z + (quantized - z).detach()
        return indices, quantized, loss

    def rotate_dead_codes(self, z: Tensor, *, dead_threshold: int = 10) -> int:
        """Move unused (or rarely used) codes to random live input points.

        Zeghidour 2022: VQ training is brittle when some codes stop being
        assigned. This method finds codes whose usage_counter <= dead_threshold
        and replaces them with randomly-sampled rows from z.

        Returns the number of codes rotated.
        """
        with torch.no_grad():
            dead_mask = self.usage_counter <= dead_threshold
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0 or z.shape[0] == 0:
                return 0
            idx = torch.randint(0, z.shape[0], (n_dead,))
            new_embeds = z[idx].detach().clone()
            if self.ema:
                self.embeddings = self.embeddings.clone()
                self.embeddings[dead_mask] = new_embeds
                self.ema_embed_sum[dead_mask]    = new_embeds
                self.ema_cluster_size[dead_mask] = 1.0
            else:
                self.embeddings.data[dead_mask] = new_embeds
            self.usage_counter[dead_mask] = 0
            return n_dead
