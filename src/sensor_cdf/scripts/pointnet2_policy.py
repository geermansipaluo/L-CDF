#!/usr/bin/env python3
"""
PointNet++ policy baseline for L-CDF / DensityNet experiments.

This file ports the core PointNet++ Set Abstraction idea from the official
TensorFlow implementation into a self-contained PyTorch module that can be
dropped into the current training pipeline.

Official PointNet++ reference structure:
  - sample_and_group
  - pointnet_sa_module
  - hierarchical set abstraction layers
  - global feature + fully-connected head

Adaptation here:
  - input point cloud is 2D local LiDAR points [x, y], not 3D object points;
  - variable-size PyG batches are converted to dense [B, N, 2] tensors;
  - output is a continuous control input [v, L*omega];
  - forward signature matches UNet.forward(state, points, G_cdf, h_cdf), so
    trainer.py can switch architectures with minimal changes.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class TimingMixin:
    def _init_timing(self, enable_timing_debug: bool = False, timing_sync_cuda: bool = True):
        self.enable_timing_debug = bool(enable_timing_debug)
        self.timing_sync_cuda = bool(timing_sync_cuda)
        self._timing_stats = {}

    def _timing_now(self, device=None):
        if self.enable_timing_debug and self.timing_sync_cuda and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device=device)
            except Exception:
                torch.cuda.synchronize()
        return time.perf_counter()

    def _timing_add(self, name: str, start_time: float, device=None):
        if not getattr(self, "enable_timing_debug", False):
            return
        if self.timing_sync_cuda and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device=device)
            except Exception:
                torch.cuda.synchronize()
        dt = time.perf_counter() - start_time
        stat = self._timing_stats.setdefault(name, [0.0, 0])
        stat[0] += float(dt)
        stat[1] += 1

    def reset_timing_stats(self):
        self._timing_stats = {}

    def get_timing_report(self, prefix: str = ""):
        return {
            prefix + k: {
                "total": float(total),
                "count": int(count),
                "avg": float(total) / max(int(count), 1),
            }
            for k, (total, count) in getattr(self, "_timing_stats", {}).items()
        }


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between each pair of points.

    Args:
        src: [B, N, C]
        dst: [B, M, C]

    Returns:
        dist: [B, N, M]
    """
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points with batched indices.

    Args:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]

    Returns:
        gathered: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling.

    Args:
        xyz: [B, N, C]
        npoint: number of sampled points

    Returns:
        centroids: [B, npoint]
    """
    B, N, _ = xyz.shape
    npoint = max(1, min(int(npoint), int(N)))
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices

    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query with deterministic fallback to nearest points if the radius
    contains too few neighbors.

    Args:
        radius: local region radius
        nsample: maximum number of neighbors
        xyz: [B, N, C]
        new_xyz: [B, S, C]

    Returns:
        group_idx: [B, S, nsample]
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    nsample = max(1, min(int(nsample), int(N)))

    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    sorted_idx = torch.argsort(sqrdists, dim=-1)[:, :, :nsample]  # [B, S, nsample]
    sorted_dist = torch.gather(sqrdists, dim=2, index=sorted_idx)

    if radius is None or float(radius) <= 0.0:
        return sorted_idx

    radius2 = float(radius) ** 2
    within = sorted_dist <= radius2
    first_idx = sorted_idx[:, :, 0:1].expand(B, S, nsample)
    group_idx = torch.where(within, sorted_idx, first_idx)
    return group_idx


class SharedMLP2d(nn.Module):
    """1x1 Conv2d + BatchNorm + ReLU stack used inside Set Abstraction."""

    def __init__(self, channels):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointNetSetAbstraction(nn.Module):
    """
    PyTorch PointNet++ Set Abstraction layer.

    Compared with the official TF module:
      - uses PyTorch tensor ops instead of custom TF sampling/grouping ops;
      - supports 2D local LiDAR points directly;
      - uses max pooling over each local neighborhood.
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp,
        group_all: bool = False,
        use_xyz: bool = True,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = bool(group_all)
        self.use_xyz = bool(use_xyz)

        mlp_in = int(in_channel)
        self.mlp = SharedMLP2d([mlp_in] + list(mlp))

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [B, N, C]
            points: [B, N, D] or None

        Returns:
            new_xyz: [B, S, C]
            new_points: [B, S, D_out]
        """
        B, N, C = xyz.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 1, C, device=xyz.device, dtype=xyz.dtype)
            grouped_xyz = xyz.view(B, 1, N, C)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
                if self.use_xyz:
                    new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz
        else:
            npoint = max(1, min(int(self.npoint), int(N)))
            nsample = max(1, min(int(self.nsample), int(N)))
            fps_idx = farthest_point_sample(xyz, npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(float(self.radius), nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx)
                if self.use_xyz:
                    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz_norm

        # [B, S, K, D] -> [B, D, K, S]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2).values  # [B, D_out, S]
        new_points = new_points.transpose(1, 2).contiguous()  # [B, S, D_out]
        return new_xyz, new_points


class PointNet2Encoder2D(TimingMixin, nn.Module):
    """
    PointNet++ encoder for 2D local LiDAR point sets.
    """

    def __init__(
        self,
        max_points: int = 200,
        point_dim: int = 2,
        hidden_dim: int = 256,
        npoint1: int = 64,
        radius1: float = 0.5,
        nsample1: int = 16,
        npoint2: int = 16,
        radius2: float = 1.0,
        nsample2: int = 16,
        padding_value: float = 99.0,
        enable_timing_debug: bool = False,
        timing_sync_cuda: bool = True,
    ):
        super().__init__()
        TimingMixin._init_timing(self, enable_timing_debug=enable_timing_debug, timing_sync_cuda=timing_sync_cuda)
        self.max_points = int(max_points)
        self.point_dim = int(point_dim)
        self.padding_value = float(padding_value)

        # Input points initially have no extra per-point feature, so SA1 uses xyz only.
        self.sa1 = PointNetSetAbstraction(
            npoint=npoint1,
            radius=radius1,
            nsample=nsample1,
            in_channel=self.point_dim,
            mlp=[64, 64, 128],
            group_all=False,
            use_xyz=True,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=npoint2,
            radius=radius2,
            nsample=nsample2,
            in_channel=self.point_dim + 128,
            mlp=[128, 128, 256],
            group_all=False,
            use_xyz=True,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=self.point_dim + 256,
            mlp=[256, 512, hidden_dim],
            group_all=True,
            use_xyz=True,
        )

    def _fixed_size_points(self, pos: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert PyG sparse points to dense fixed-size [B, max_points, point_dim].
        Invalid/padding points with value 99 are removed before padding.
        """
        device = pos.device
        dtype = pos.dtype

        pos = pos[:, : self.point_dim].to(device=device, dtype=dtype)
        pts_dense, mask = to_dense_batch(
            pos,
            batch_idx,
            fill_value=float(self.padding_value),
        )

        out = []
        for b in range(pts_dense.shape[0]):
            pts_b = pts_dense[b]
            mask_b = mask[b]
            finite = torch.isfinite(pts_b).all(dim=1)
            not_padding = ~torch.all(torch.abs(pts_b - self.padding_value) < 1e-4, dim=1)
            valid = mask_b & finite & not_padding
            pts_valid = pts_b[valid]

            if pts_valid.numel() == 0:
                pts_valid = torch.zeros(1, self.point_dim, device=device, dtype=dtype)

            if pts_valid.shape[0] >= self.max_points:
                pts_fixed = pts_valid[: self.max_points]
            else:
                repeat_count = self.max_points - pts_valid.shape[0]
                pad = pts_valid[-1:, :].repeat(repeat_count, 1)
                pts_fixed = torch.cat([pts_valid, pad], dim=0)

            out.append(pts_fixed)

        return torch.stack(out, dim=0)

    def forward(self, points) -> torch.Tensor:
        """
        Args:
            points: PyG Data/Batch with pos [Total,2] and batch [Total]

        Returns:
            global feature: [B, hidden_dim]
        """
        t_total = self._timing_now(device=points.pos.device)
        xyz = self._fixed_size_points(points.pos, points.batch)
        xyz = xyz[:, :, : self.point_dim].contiguous()

        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        feat = l3_points.squeeze(1)

        self._timing_add("pointnet2/encoder_total", t_total, device=points.pos.device)
        return feat


class PointNet2Policy(TimingMixin, nn.Module):
    """
    PointNet++ behavior cloning policy baseline.

    It intentionally does not include the CDF-QP safety layer. The policy directly
    outputs the executed control input. forward() returns (u_safe, u_nom) for
    compatibility with the existing trainer, but both are the same tensor.
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 256,
        max_points: int = 200,
        output_scale: float = 1.2,
        point_dim: int = 2,
        npoint1: int = 64,
        radius1: float = 0.5,
        nsample1: int = 16,
        npoint2: int = 16,
        radius2: float = 1.0,
        nsample2: int = 16,
        padding_value: float = 99.0,
        enable_timing_debug: bool = False,
        timing_sync_cuda: bool = True,
    ):
        super().__init__()
        TimingMixin._init_timing(self, enable_timing_debug=enable_timing_debug, timing_sync_cuda=timing_sync_cuda)
        self.output_scale = float(output_scale)
        self.use_safety_layer = False
        self.use_learned_cdf_constraints = False
        self.ablation = "pointnet2_bc"

        self.point_encoder = PointNet2Encoder2D(
            max_points=max_points,
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            npoint1=npoint1,
            radius1=radius1,
            nsample1=nsample1,
            npoint2=npoint2,
            radius2=radius2,
            nsample2=nsample2,
            padding_value=padding_value,
            enable_timing_debug=enable_timing_debug,
            timing_sync_cuda=timing_sync_cuda,
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )

    def forward_nominal(self, state, points):
        t_total = self._timing_now(device=state.device)
        point_feat = self.point_encoder(points)
        state_feat = self.state_encoder(state)
        fused = self.fusion(torch.cat([point_feat, state_feat], dim=1))
        u = self.head(fused) * self.output_scale
        self._timing_add("pointnet2/policy_forward", t_total, device=state.device)
        return u

    def forward(self, state, points, G_cdf=None, h_cdf=None):
        u = self.forward_nominal(state, points)
        return u, u

    def cdf_parameter_dict(self):
        return {}

    def reset_timing_stats(self):
        TimingMixin.reset_timing_stats(self)
        if hasattr(self.point_encoder, "reset_timing_stats"):
            self.point_encoder.reset_timing_stats()

    def get_timing_report(self):
        report = TimingMixin.get_timing_report(self, prefix="")
        if hasattr(self.point_encoder, "get_timing_report"):
            report.update(self.point_encoder.get_timing_report(prefix=""))
        return report
