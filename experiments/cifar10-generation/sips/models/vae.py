import math

import torch
from torch import nn


LOGIT_LAMBDA = 0.05


class MLPVAE(nn.Module):
    """Fully-connected Bernoulli VAE for small grayscale images."""

    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = 16,
        in_channels: int = 1,
        hidden_dims: tuple = (512, 256),
    ) -> None:
        super().__init__()
        self.img_dim = img_dim
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)

        input_shape = in_channels * img_dim * img_dim

        encoder = [nn.Flatten()]
        prev = input_shape
        for h in self.hidden_dims:
            encoder += [nn.Linear(prev, h), nn.ReLU(True)]
            prev = h
        self.encoder = nn.Sequential(*encoder)
        self.mu_head = nn.Linear(prev, latent_dim)
        self.logvar_head = nn.Linear(prev, latent_dim)

        decoder = []
        prev = latent_dim
        for h in reversed(self.hidden_dims):
            decoder += [nn.Linear(prev, h), nn.ReLU(True)]
            prev = h
        decoder += [
            nn.Linear(prev, input_shape),
            nn.Sigmoid(),
            nn.Unflatten(1, (in_channels, img_dim, img_dim)),
        ]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_mean(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_mean(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode_mean(z)


def bernoulli_vae_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict:
    """ELBO for a Bernoulli-decoder VAE, summed over pixels & batch."""
    bce = nn.functional.binary_cross_entropy(x_rec, x, reduction="none")
    rec = bce.sum(dim=tuple(range(1, x.ndim))).sum()
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    return {"vae": rec + beta * kl, "rec": rec, "kl": kl}


class DCVAE(nn.Module):
    """Diagonal-Gaussian conv VAE (32x32 RGB) operating in logit pixel space."""

    def __init__(
        self,
        img_dim: int = 32,
        latent_dim: int = 64,
        in_channels: int = 3,
        num_feature: int = 32,
        deep: bool = False,
        sigma_vae: bool = False,
    ) -> None:
        super().__init__()
        self.img_dim = img_dim
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_feature = num_feature
        self.deep = deep
        self.sigma_vae = sigma_vae
        out_channels = in_channels if sigma_vae else 2 * in_channels

        C, F, L = in_channels, num_feature, latent_dim

        def enc_refine(c):
            return [
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        def dec_refine(c):
            return [
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(True),
            ]

        enc = [
            nn.Conv2d(C, F, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if deep:
            enc += enc_refine(F)
        for in_c, out_c in [(F, 2 * F), (2 * F, 4 * F), (4 * F, 8 * F)]:
            enc += [
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if deep:
                enc += enc_refine(out_c)
        enc += [nn.Flatten()]
        self.encoder = nn.Sequential(*enc)

        prev_dim = F * 8 * 2 * 2
        self.mu_head = nn.Linear(prev_dim, L)
        self.logvar_head = nn.Linear(prev_dim, L)

        dec = [
            nn.Unflatten(1, (L, 1, 1)),
            nn.ConvTranspose2d(L, F * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(F * 8),
            nn.ReLU(True),
        ]
        if deep:
            dec += dec_refine(F * 8)
        for in_c, out_c in [(8 * F, 4 * F), (4 * F, 2 * F), (2 * F, F)]:
            dec += [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            ]
            if deep:
                dec += dec_refine(out_c)
        dec += [nn.ConvTranspose2d(F, out_channels, 4, 2, 1, bias=False)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_full(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode_mean(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z)
        return y if self.sigma_vae else y[:, : self.in_channels]

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_full(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        y = self.decoder(z)
        if self.sigma_vae:
            return y
        C = self.in_channels
        mu = y[:, :C]
        logvar = torch.clamp(y[:, C:], min=-7.0)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def gaussian_vae_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    in_channels: int,
    beta: float = 1.0,
) -> dict:
    """ELBO for a diagonal-Gaussian-decoder VAE in logit space.

    Expects ``x`` pre-mapped to logit space (via ``LogitTransform``) and
    ``y`` to be the decoder output with ``2*in_channels`` channels stacking
    mean and log-variance.  Log-variance is clamped at -7 for stability.
    """
    C = in_channels
    D = x.shape[1] * x.shape[2] * x.shape[3]
    mu_theta = y[:, :C]
    logvar_theta = torch.clamp(y[:, C:], min=-7.0)
    inv_std = torch.exp(-0.5 * logvar_theta)
    sse = torch.sum((inv_std * (x - mu_theta)) ** 2, dim=(1, 2, 3))
    logpxz = -0.5 * (
        D * math.log(2.0 * math.pi)
        + logvar_theta.sum(dim=(1, 2, 3))
        + sse
    )
    rec = -logpxz.sum()
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    return {"vae": rec + beta * kl, "rec": rec, "kl": kl}


def sigma_vae_loss(
    x: torch.Tensor,
    mu_theta: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    sigma_min: float = math.exp(-6.0),
) -> dict:
    """Optimal-sigma VAE loss (Rybkin et al. 2020, Eq. 6+8).

    Decoder outputs the Gaussian mean only; sigma is computed analytically
    from the batch MSE each step, giving a calibrated decoder without
    learning a logvar head or tuning beta.
    """
    N = x.shape[0]
    D = x[0].numel()
    mse = torch.mean((x - mu_theta) ** 2)
    mse = torch.clamp(mse, min=sigma_min ** 2)
    rec = 0.5 * N * D * torch.log(mse)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    return {"vae": rec + kl, "rec": rec, "kl": kl, "sigma": mse.sqrt()}


class UniformDequantization:
    """Add uniform noise within each 1/256 quantization bin."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (torch.rand_like(t) + t * 255.0) / 256.0


class LogitTransform:
    """Map [0, 1] pixels to unbounded reals via logit(lambda + (1-2*lambda)*x)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logit(LOGIT_LAMBDA + (1.0 - 2.0 * LOGIT_LAMBDA) * x)


def inverse_logit_transform(y: torch.Tensor) -> torch.Tensor:
    """Invert LogitTransform back to [0, 1] pixels."""
    return (torch.sigmoid(y) - LOGIT_LAMBDA) / (1.0 - 2.0 * LOGIT_LAMBDA)
