"""Unconditional convolutional HINT normalizing flow.

Each tree squeezes the input, splits channels, processes the upper half
recursively, uses it to predict the affine coupling parameters ``(S, T)``
for the lower half (``S = exp(clamp(s, -5, 5))``), then processes the
lower half recursively. ``n_flow_layers`` trees are stacked with random
pixel permutations between them.

The base distribution is a standard normal in pixel space, so

    log p(x) = -0.5 * ||z||^2 - 0.5 * d * log(2 pi) + log|det J|

where ``d = C * H * W`` and ``log|det J|`` is the sum of ``log_S`` over
all spatial / channel locations accumulated through the tree.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _squeeze(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, 4C, H/2, W/2]."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, 4 * C, H // 2, W // 2)
    return x


def _unsqueeze(x: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
    """[B, 4C, H/2, W/2] -> [B, C, H, W]."""
    B, C4, Hh, Wh = x.shape
    C = C4 // 4
    x = x.reshape(B, C, 2, 2, Hh, Wh)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, target_H, target_W)
    return x


class _ConvNet(nn.Module):
    """Three-conv coupling subnet producing (log_S, T) for the lower channels."""

    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 32) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_ch, 2 * out_ch, 3, padding=1)
        nn.init.normal_(self.conv3.weight, std=0.01)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x_upper: torch.Tensor) -> tuple:
        h = F.silu(self.conv1(x_upper))
        h = F.silu(self.conv2(h))
        out = self.conv3(h)
        return out[:, : self.out_ch], out[:, self.out_ch :]


class _ConvNetMLP(nn.Module):
    """MLP coupling subnet for the 1x1 spatial base case."""

    def __init__(self, in_ch: int, out_ch: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_dim, 2 * out_ch)
        nn.init.normal_(self.final.weight, std=0.01)
        nn.init.zeros_(self.final.bias)

    def forward(self, x_upper: torch.Tensor) -> tuple:
        B = x_upper.shape[0]
        h = self.net(x_upper.reshape(B, -1))
        out = self.final(h)
        S = out[:, : self.out_ch].reshape(B, self.out_ch, 1, 1)
        T = out[:, self.out_ch :].reshape(B, self.out_ch, 1, 1)
        return S, T


class _HINTTree(nn.Module):
    """Recursive squeeze-coupling tree (single HINT block)."""

    def __init__(self, in_ch: int, spatial_size: int, hidden_ch: int = 32) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.spatial_size = spatial_size

        if spatial_size < 2 or spatial_size % 2 != 0:
            self.is_leaf = True
            return
        self.is_leaf = False

        squeezed_ch = 4 * in_ch
        squeezed_spatial = spatial_size // 2
        self.upper_ch = squeezed_ch // 2
        self.lower_ch = squeezed_ch - self.upper_ch

        if squeezed_spatial <= 1:
            self.coupling = _ConvNetMLP(
                self.upper_ch, self.lower_ch, hidden_dim=hidden_ch * 2,
            )
        else:
            self.coupling = _ConvNet(
                self.upper_ch, self.lower_ch, hidden_ch=hidden_ch,
            )

        self.upper_tree = _HINTTree(self.upper_ch, squeezed_spatial, hidden_ch)
        self.lower_tree = _HINTTree(self.lower_ch, squeezed_spatial, hidden_ch)

    def _coupling_params(self, z_upper: torch.Tensor) -> tuple:
        S_raw, T = self.coupling(z_upper)
        log_S = torch.clamp(S_raw, -5.0, 5.0)
        return torch.exp(log_S), T, log_S

    def forward(self, x: torch.Tensor) -> tuple:
        if self.is_leaf:
            return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        _, _, H, W = x.shape
        x_sq = _squeeze(x)
        x_upper = x_sq[:, : self.upper_ch]
        x_lower = x_sq[:, self.upper_ch :]

        z_upper, ld_upper = self.upper_tree(x_upper)
        S, T, log_S = self._coupling_params(z_upper)
        x_lower_coupled = S * x_lower + T
        z_lower, ld_lower = self.lower_tree(x_lower_coupled)

        z_sq = torch.cat([z_upper, z_lower], dim=1)
        z = _unsqueeze(z_sq, H, W)
        log_det = ld_upper + log_S.sum(dim=(1, 2, 3)) + ld_lower
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_leaf:
            return z

        _, _, H, W = z.shape
        z_sq = _squeeze(z)
        z_upper = z_sq[:, : self.upper_ch]
        z_lower = z_sq[:, self.upper_ch :]

        x_upper = self.upper_tree.inverse(z_upper)
        S, T, _ = self._coupling_params(z_upper)
        x_lower_coupled = self.lower_tree.inverse(z_lower)
        x_lower = (x_lower_coupled - T) / S

        x_sq = torch.cat([x_upper, x_lower], dim=1)
        return _unsqueeze(x_sq, H, W)


class UnconditionalConvHINT(nn.Module):
    """Stack of ``n_flow_layers`` :class:`_HINTTree`s with pixel permutations.

    Args:
        in_channels: Number of image channels (1 for the Parihaka patches).
        spatial_size: Spatial resolution. Must be a power of 2 for full
            recursion down to 1x1.
        hidden_ch: Hidden channels inside each conv coupling subnet.
        n_flow_layers: Number of stacked trees.
    """

    def __init__(
        self,
        in_channels: int = 1,
        spatial_size: int = 128,
        hidden_ch: int = 32,
        n_flow_layers: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.n_pixels = spatial_size * spatial_size

        self.trees = nn.ModuleList()
        perms, inv_perms = [], []
        for _ in range(n_flow_layers):
            self.trees.append(_HINTTree(in_channels, spatial_size, hidden_ch))
            perm = torch.randperm(self.n_pixels)
            perms.append(perm)
            inv_perms.append(torch.argsort(perm))
        self.register_buffer("perms", torch.stack(perms))
        self.register_buffer("inv_perms", torch.stack(inv_perms))

    def _permute(self, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W)[:, :, perm].reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> tuple:
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for k, tree in enumerate(self.trees):
            z = self._permute(z, self.perms[k])
            z, ld = tree(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for k in reversed(range(len(self.trees))):
            x = self.trees[k].inverse(x)
            x = self._permute(x, self.inv_perms[k])
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        d = self.in_channels * self.n_pixels
        log_pz = -0.5 * z.pow(2).sum(dim=(1, 2, 3)) - 0.5 * d * math.log(2 * math.pi)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        self.eval()
        z = torch.randn(
            num_samples, self.in_channels, self.spatial_size, self.spatial_size,
            device=device,
        )
        x = self.inverse(z)
        return x.cpu()
