from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class HGPLeaf:
    xmin: float
    xmax: float
    zmin: float
    zmax: float
    depth: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) * 0.5, (self.zmin + self.zmax) * 0.5)

    @property
    def size(self) -> Tuple[float, float]:
        return (max(self.xmax - self.xmin, 1e-8), max(self.zmax - self.zmin, 1e-8))


class HLCUController:

    def __init__(
        self,
        warmup_iter: int = 2000,
        update_interval: int = 1000,
        max_depth: int = 5,
        min_points: int = 4000,
        max_zones: int = 32,
        overlap_ratio: float = 0.08,
        loss_weight: float = 0.002,
        sinkhorn_eps: float = 0.08,
        sinkhorn_iters: int = 5,
        max_points_per_zone: int = 48,
        tgc_start_iter: int = 3000,
        tgc_interval: int = 20,
        tgc_max_pairs: int = 8,
        device: str = "cuda",
    ):
        self.warmup_iter = int(warmup_iter)
        self.update_interval = int(update_interval)
        self.max_depth = int(max_depth)
        self.min_points = int(min_points)
        self.max_zones = int(max_zones)
        self.overlap_ratio = float(overlap_ratio)
        self.loss_weight = float(loss_weight)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.max_points_per_zone = int(max_points_per_zone)
        self.tgc_start_iter = int(tgc_start_iter)
        self.tgc_interval = max(int(tgc_interval), 1)
        self.tgc_max_pairs = max(int(tgc_max_pairs), 1)
        self.device = device

        self.leaves: List[HGPLeaf] = []
        self.adjacent_pairs: List[Tuple[int, int]] = []
        self.last_update_iter: int = -1
        self._last_count: int = -1

    @staticmethod
    def _safe_norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if hi <= lo + 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _compute_heterogeneity_scores(self, xyz_np: np.ndarray, grad_np: np.ndarray) -> np.ndarray:
        x, z = xyz_np[:, 0], xyz_np[:, 2]
        grad_score = self._safe_norm(grad_np.reshape(-1))

        bins = int(np.clip(np.sqrt(max(len(x), 1)) / 3, 16, 96))
        hist, xedges, zedges = np.histogram2d(x, z, bins=bins)
        xi = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, bins - 1)
        zi = np.clip(np.searchsorted(zedges, z, side="right") - 1, 0, bins - 1)
        density = hist[xi, zi]
        sparse_score = 1.0 - self._safe_norm(density)

                                                         
        return (0.65 * grad_score + 0.35 * sparse_score).astype(np.float32)

    def _recursive_split(
        self,
        xyz_np: np.ndarray,
        score_np: np.ndarray,
        indices: np.ndarray,
        depth: int,
        leaves: List[np.ndarray],
    ) -> None:
        if depth >= self.max_depth or len(indices) <= self.min_points or len(leaves) >= self.max_zones:
            leaves.append(indices)
            return
        if len(indices) < 2 * self.min_points:
            leaves.append(indices)
            return

        pts = xyz_np[indices]
        scores = score_np[indices]
        w = scores + 1e-3
        w = w / (w.sum() + 1e-8)

                                                                                
        axes = [0, 2]
        weighted_vars = []
        for ax in axes:
            coord = pts[:, ax]
            mean = (coord * w).sum()
            weighted_vars.append(((coord - mean) ** 2 * w).sum())
        ax = axes[int(np.argmax(weighted_vars))]

        coord = pts[:, ax]
        order = np.argsort(coord)
        csum = np.cumsum(w[order])
        mid_pos = int(np.searchsorted(csum, 0.5))
        split_value = coord[order[np.clip(mid_pos, 1, len(order) - 2)]]
        left = indices[coord <= split_value]
        right = indices[coord > split_value]

        if len(left) < self.min_points // 2 or len(right) < self.min_points // 2:
            median = np.median(coord)
            left = indices[coord <= median]
            right = indices[coord > median]

        if len(left) == 0 or len(right) == 0:
            leaves.append(indices)
            return

        self._recursive_split(xyz_np, score_np, left, depth + 1, leaves)
        self._recursive_split(xyz_np, score_np, right, depth + 1, leaves)

    @staticmethod
    def _bbox_gap(a: HGPLeaf, b: HGPLeaf) -> float:
                                                                     
        dx = max(a.xmin - b.xmax, b.xmin - a.xmax, 0.0)
        dz = max(a.zmin - b.zmax, b.zmin - a.zmax, 0.0)
        return float((dx * dx + dz * dz) ** 0.5)

    def _build_adjacent_pairs(self) -> None:
        pairs = []
        if len(self.leaves) < 2:
            self.adjacent_pairs = []
            return
        diag = 0.0
        for l in self.leaves:
            sx, sz = l.size
            diag = max(diag, (sx * sx + sz * sz) ** 0.5)
        gap_thr = max(diag * self.overlap_ratio * 1.5, 1e-8)
        for i in range(len(self.leaves)):
            for j in range(i + 1, len(self.leaves)):
                gap = self._bbox_gap(self.leaves[i], self.leaves[j])
                if gap <= gap_thr:
                    pairs.append((i, j))
                                                                                                
        if not pairs:
            centers = np.array([l.center for l in self.leaves], dtype=np.float32)
            d = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            for i in range(len(self.leaves)):
                nn = np.argsort(d[i])[1:3]
                for j in nn:
                    a, b = sorted((i, int(j)))
                    if a != b and (a, b) not in pairs:
                        pairs.append((a, b))
        self.adjacent_pairs = pairs

    def update_partitions(self, gaussians, iteration: int) -> None:
        if iteration < self.warmup_iter:
            return
        if self.leaves and (iteration - self.last_update_iter) < self.update_interval:
            return

        with torch.no_grad():
            xyz = gaussians.get_xyz.detach()
            if xyz.numel() == 0:
                return
            xyz_np = xyz.float().cpu().numpy()
            if hasattr(gaussians, "xyz_gradient_accum") and hasattr(gaussians, "denom"):
                grad = gaussians.xyz_gradient_accum / torch.clamp_min(gaussians.denom, 1.0)
                grad_np = grad.detach().float().cpu().numpy().reshape(-1)
                grad_np[~np.isfinite(grad_np)] = 0.0
            else:
                grad_np = np.zeros((xyz_np.shape[0],), dtype=np.float32)

        score_np = self._compute_heterogeneity_scores(xyz_np, grad_np)
        index_leaves: List[np.ndarray] = []
        self._recursive_split(xyz_np, score_np, np.arange(xyz_np.shape[0]), 0, index_leaves)

        leaves: List[HGPLeaf] = []
        eps = 1e-6
        for idx in index_leaves[: self.max_zones]:
            pts = xyz_np[idx]
            if pts.shape[0] == 0:
                continue
            x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
            z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
            pad_x = max((x1 - x0) * 0.01, eps)
            pad_z = max((z1 - z0) * 0.01, eps)
            leaves.append(HGPLeaf(x0 - pad_x, x1 + pad_x, z0 - pad_z, z1 + pad_z, 0))

        self.leaves = leaves
        self._build_adjacent_pairs()
        self.last_update_iter = iteration
        self._last_count = xyz_np.shape[0]
        print(
            f"[HLCU] HGP updated at iter={iteration}: "
            f"zones={len(self.leaves)}, pairs={len(self.adjacent_pairs)}, gaussians={xyz_np.shape[0]}"
        )

    def _assign_zones_current(self, xyz: torch.Tensor) -> torch.Tensor:
        n = xyz.shape[0]
        if not self.leaves:
            return torch.full((n,), -1, dtype=torch.long, device=xyz.device)
        xz = xyz.detach()[:, [0, 2]]
        centers = torch.tensor([l.center for l in self.leaves], device=xyz.device, dtype=xz.dtype)
        boxes = torch.tensor([[l.xmin, l.xmax, l.zmin, l.zmax] for l in self.leaves], device=xyz.device, dtype=xz.dtype)
        inside = (xz[:, 0:1] >= boxes[:, 0]) & (xz[:, 0:1] <= boxes[:, 1]) & (xz[:, 1:2] >= boxes[:, 2]) & (xz[:, 1:2] <= boxes[:, 3])
        d = ((xz[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                                                                                            
        d = torch.where(inside, d, d + 1e6)
        zids = d.argmin(dim=1)
        no_inside = ~inside.any(dim=1)
        if no_inside.any():
            d2 = ((xz[no_inside, None, :] - centers[None, :, :]) ** 2).sum(-1)
            zids[no_inside] = d2.argmin(dim=1)
        return zids

    def _sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
                                                                    
        cost = cost - cost.min()
        K = torch.exp(-cost / max(self.sinkhorn_eps, 1e-6)).clamp_min(1e-8)
        m, n = K.shape
        u = torch.ones(m, device=K.device, dtype=K.dtype) / max(m, 1)
        v = torch.ones(n, device=K.device, dtype=K.dtype) / max(n, 1)
        a = torch.full((m,), 1.0 / max(m, 1), device=K.device, dtype=K.dtype)
        b = torch.full((n,), 1.0 / max(n, 1), device=K.device, dtype=K.dtype)
        for _ in range(self.sinkhorn_iters):
            u = a / (K @ v + 1e-8)
            v = b / (K.t() @ u + 1e-8)
        P = u[:, None] * K * v[None, :]
        return P / (P.sum() + 1e-8)

    def _sample_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.numel() <= self.max_points_per_zone:
            return idx
        perm = torch.randperm(idx.numel(), device=idx.device)[: self.max_points_per_zone]
        return idx[perm]

    def _choose_pair_subset(self, xyz: torch.Tensor, zids: torch.Tensor) -> List[Tuple[int, int]]:
        if len(self.adjacent_pairs) <= self.tgc_max_pairs:
            return list(self.adjacent_pairs)
                                                                                      
        pairs = list(self.adjacent_pairs)
        if len(pairs) == 0:
            return []
        if np.random.rand() < 0.5:
            choice = np.random.choice(len(pairs), size=self.tgc_max_pairs, replace=False)
            return [pairs[int(i)] for i in choice]
        centers = []
        for l in self.leaves:
            centers.append(l.center)
        centers = torch.tensor(centers, device=xyz.device, dtype=xyz.dtype)
        vals = []
        for k, (a, b) in enumerate(pairs):
            vals.append((torch.sum((centers[a] - centers[b]) ** 2).item(), k))
        vals.sort(key=lambda x: x[0])
        return [pairs[k] for _, k in vals[: self.tgc_max_pairs]]

    def tgc_loss(self, gaussians, iteration: Optional[int] = None) -> torch.Tensor:
        xyz = gaussians.get_xyz
        zero = xyz.new_tensor(0.0)
        if not self.leaves or len(self.leaves) < 2 or not self.adjacent_pairs:
            return zero
        if iteration is not None:
            if iteration < self.tgc_start_iter:
                return zero
            if iteration % self.tgc_interval != 0:
                return zero

        zids = self._assign_zones_current(xyz)
        if (zids >= 0).sum() < 8:
            return zero

        with torch.no_grad():
            extent = torch.quantile(xyz.detach(), 0.95, dim=0) - torch.quantile(xyz.detach(), 0.05, dim=0)
            scene_scale = torch.clamp(torch.norm(extent[[0, 2]]), min=1e-4)
            radius = scene_scale * self.overlap_ratio

        fdc = gaussians.get_features_dc.squeeze(1)
        scaling = gaussians.get_scaling
        opacity = gaussians.get_opacity

        total = zero
        pairs_used = 0
        pairs = self._choose_pair_subset(xyz, zids)

        for za, zb in pairs:
            idx_a_all = torch.nonzero(zids == za, as_tuple=False).flatten()
            idx_b_all = torch.nonzero(zids == zb, as_tuple=False).flatten()
            if idx_a_all.numel() < 4 or idx_b_all.numel() < 4:
                continue

                                                                                   
            ca = torch.tensor(self.leaves[za].center, device=xyz.device, dtype=xyz.dtype)
            cb = torch.tensor(self.leaves[zb].center, device=xyz.device, dtype=xyz.dtype)
            da = torch.norm(xyz[idx_a_all].detach()[:, [0, 2]] - cb[None, :], dim=1)
            db = torch.norm(xyz[idx_b_all].detach()[:, [0, 2]] - ca[None, :], dim=1)
            ka = min(max(self.max_points_per_zone, 4), idx_a_all.numel())
            kb = min(max(self.max_points_per_zone, 4), idx_b_all.numel())
            idx_a = idx_a_all[torch.topk(da, k=ka, largest=False).indices]
            idx_b = idx_b_all[torch.topk(db, k=kb, largest=False).indices]
            idx_a = self._sample_indices(idx_a)
            idx_b = self._sample_indices(idx_b)
            if idx_a.numel() < 4 or idx_b.numel() < 4:
                continue

            xa, xb = xyz[idx_a], xyz[idx_b]
            pos_cost = torch.cdist(xa, xb).pow(2) / (radius ** 2 + 1e-8)
                                                                                                 
            with torch.no_grad():
                color_cost = torch.cdist(fdc[idx_a].detach(), fdc[idx_b].detach(), p=1)
                scale_cost = torch.cdist(torch.log(scaling[idx_a].detach() + 1e-8), torch.log(scaling[idx_b].detach() + 1e-8), p=2)
                cost = pos_cost.detach() + 0.05 * color_cost + 0.02 * scale_cost
                P = self._sinkhorn(cost)

            sc_agree = torch.cdist(torch.log(scaling[idx_a] + 1e-8), torch.log(scaling[idx_b] + 1e-8), p=2).pow(2)
            col_agree = torch.cdist(fdc[idx_a], fdc[idx_b], p=1)
            op_agree = torch.cdist(opacity[idx_a], opacity[idx_b], p=1)
            pair_loss = (P * (pos_cost + 0.02 * sc_agree + 0.02 * col_agree + 0.01 * op_agree)).sum()
            total = total + pair_loss
            pairs_used += 1

        if pairs_used == 0:
            return zero
        return self.loss_weight * total / pairs_used
