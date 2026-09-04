"""DDPM training/sampling wrapper around ``diffusers.UNet2DModel``.

A UNet2DModel score net, a DDPMScheduler, an EMA copy with decay 0.9999,
and Adam with linear-warmup LR. The class owns the model, scheduler,
optimizer, LR scheduler, and EMA state; scripts handle the data loaders,
checkpointing, and plotting.
"""

from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

EMA_DECAY = 0.9999


class SeismicDDPM(nn.Module):
    """DDPM with EMA and linear-warmup LR on a UNet2DModel score network.

    Constructor expects a ``projorg``-style ``args`` namespace with:
        image_size, block_channels (sequence), block_nlayers, attn_dim,
        dropout, nt, beta_schedule, lr, warmup_steps.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.image_size = args.image_size

        attn_dim = args.attn_dim if args.attn_dim > 0 else None
        self.model = UNet2DModel(
            in_channels=1,
            out_channels=1,
            sample_size=(self.image_size, self.image_size),
            block_out_channels=tuple(args.block_channels),
            layers_per_block=args.block_nlayers,
            attention_head_dim=attn_dim,
            dropout=args.dropout,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.ema_model = deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.nt,
            beta_schedule=args.beta_schedule,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, betas=(0.9, 0.999),
        )
        warmup_steps = max(1, int(args.warmup_steps))

        def warmup(step: int) -> float:
            return min(1.0, step / warmup_steps)

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup)

    def _ema_step(self) -> None:
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)

    def _step_loss(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0, len(self.noise_scheduler), (x.shape[0],), device=x.device,
        ).long()
        x_t = self.noise_scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.model(x_t, timesteps, return_dict=False)[0]
        return torch.mean((noise_pred - noise) ** 2)


    def train_epoch(
        self, 
        loader: DataLoader, 
        device: torch.device,
        only_comp_peak_mem = False
    ) -> list:
        """Run one training epoch. Returns the per-batch losses."""
        self.model.train()
        losses = []
        for (x,) in tqdm(loader, desc="train", leave=False):
            x = x.to(device)
            loss = self._step_loss(x)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self._ema_step()
            losses.append(loss.item())
            if only_comp_peak_mem:
                return losses
        return losses

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader, device: torch.device) -> float:
        """Average validation loss over the loader."""
        self.model.eval()
        total = 0.0
        n_batches = 0
        for (x,) in loader:
            x = x.to(device)
            total += self._step_loss(x).item()
            n_batches += 1
        return total / max(1, n_batches)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples via the full DDPM reverse process on the EMA model.

        Returns a CPU tensor of shape ``[num_samples, 1, H, W]`` in z-score
        space. The caller is responsible for denormalization.
        """
        self.ema_model.eval()
        scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
        )
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        x = torch.randn(
            num_samples, 1, self.image_size, self.image_size, device=device,
        )
        for t in tqdm(scheduler.timesteps, desc="sample", leave=False):
            t_batch = t.expand(num_samples).to(device)
            noise_pred = self.ema_model(x, t_batch, return_dict=False)[0]
            x = scheduler.step(noise_pred, t, x).prev_sample
        return x.cpu()

    def bundled_state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

    def load_bundled_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.ema_model.load_state_dict(state["ema"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
