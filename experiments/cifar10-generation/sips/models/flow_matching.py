import torch


def fm_linear_path(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Linear probability path ``x_t = (1 - t) * x0 + t * x1``.

    Args:
        x0: Noise tensor of shape ``(B, ...)``.
        x1: Data tensor of shape ``(B, ...)``.
        t: Time values in ``[0, 1]`` of shape ``(B,)``.
    """
    t = t.view(-1, *([1] * (x1.ndim - 1)))
    return (1.0 - t) * x0 + t * x1


@torch.no_grad()
def fm_sample(
    model: torch.nn.Module,
    shape: tuple,
    device: torch.device,
    n_steps: int = 100,
    t_scale: float = 1000.0,
    x0: torch.Tensor = None,
) -> torch.Tensor:
    """Euler integration of the flow-matching ODE from ``t=0`` to ``t=1``.

    The model is queried at ``t * t_scale`` so the time-embedding input range
    matches the integer-timestep convention used by the diffusers UNet2DModel
    (``t_scale=1000`` mirrors a DDPM schedule with ``num_train_timesteps=1000``).
    """
    x = torch.randn(shape, device=device) if x0 is None else x0.to(device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = torch.full(
            (shape[0],), i * dt * t_scale, device=device, dtype=torch.float32
        )
        v = model(x, t_val, return_dict=False)[0]
        x = x + dt * v
    return x
