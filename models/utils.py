import math
from collections import deque
import torch
from torch.nn import functional as F
import numpy as np
from data import utils as du


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[
        ..., None
    ]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_full((1,), 1e8)], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_positional_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None, t_label="t"):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} {t_label}=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


from collections import deque


class LossQueue:
    """
    Rolling statistic helper used to detect exploding losses.
    Maintains up to `maxlen` recent values as a FIFO queue.
    """

    def __init__(self, maxlen=10000, warmup=1000, explosion_ratio=100.0):
        if warmup > maxlen:
            raise ValueError("warmup cannot exceed maxlen")
        self.maxlen = maxlen
        self.warmup = warmup
        self.explosion_ratio = explosion_ratio
        self._values = deque(maxlen=maxlen)

    def __len__(self):
        return len(self._values)

    def clear(self):
        self._values.clear()

    def ready(self):
        return len(self._values) >= self.warmup

    def median(self):
        if not self._values:
            return None
        return float(np.median(self._values))

    def record(self, value):
        """
        Append a scalar value to the queue.
        Only the most recent `maxlen` values are kept.
        """
        value = float(value)
        if len(self._values) == self.maxlen:
            self._values.popleft()
        self._values.append(value)
        return value

    def record_and_check(self, value, ratio=None):
        """
        Record the latest value and return a tuple (is_exploding, median).

        Args:
            value: Scalar loss value.
            ratio: Threshold multiplier relative to the rolling median.
        """
        value = self.record(value)
        ratio = self.explosion_ratio if ratio is None else ratio
        if not self.ready():
            return False, None

        median_value = self.median()
        if median_value is None or median_value <= 0 or ratio is None or ratio <= 0:
            return False, median_value

        return value > ratio * median_value, median_value
