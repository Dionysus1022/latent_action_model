from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TruncatedDiffusionSchedule:
    """Minimal truncated diffusion schedule shared by training and runtime.

    Tensor shapes:
        betas: [num_train_steps]
        alphas: [num_train_steps]
        alpha_bars: [num_train_steps]
        alpha_bars_prev: [num_train_steps]
        sqrt_alpha_bars: [num_train_steps]
        sqrt_one_minus_alpha_bars: [num_train_steps]
        posterior_variance: [num_train_steps]
        truncation_timesteps: [truncation_steps] in descending order
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor
    truncation_timesteps: torch.Tensor
    num_train_steps: int
    truncation_steps: int
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TruncatedDiffusionSchedule":
        """Move floating-point schedule tensors to a target device/dtype."""
        return TruncatedDiffusionSchedule(
            betas=self.betas.to(device=device, dtype=dtype),
            alphas=self.alphas.to(device=device, dtype=dtype),
            alpha_bars=self.alpha_bars.to(device=device, dtype=dtype),
            alpha_bars_prev=self.alpha_bars_prev.to(device=device, dtype=dtype),
            sqrt_alpha_bars=self.sqrt_alpha_bars.to(device=device, dtype=dtype),
            sqrt_one_minus_alpha_bars=self.sqrt_one_minus_alpha_bars.to(device=device, dtype=dtype),
            posterior_variance=self.posterior_variance.to(device=device, dtype=dtype),
            truncation_timesteps=self.truncation_timesteps.to(device=device),
            num_train_steps=int(self.num_train_steps),
            truncation_steps=int(self.truncation_steps),
            beta_schedule=str(self.beta_schedule),
            beta_start=float(self.beta_start),
            beta_end=float(self.beta_end),
        )

    def step_pairs(self, include_terminal: bool = True) -> list[tuple[int, int]]:
        """Return reverse denoise step pairs.

        Example:
            truncation_timesteps = [15, 10, 5, 0]
            returns [(15, 10), (10, 5), (5, 0), (0, -1)] when include_terminal=True
        """
        timesteps = self.truncation_timesteps.detach().cpu().tolist()
        pairs = [(int(timesteps[i]), int(timesteps[i + 1])) for i in range(len(timesteps) - 1)]
        if include_terminal and len(timesteps) > 0:
            pairs.append((int(timesteps[-1]), -1))
        return pairs


def build_beta_schedule(
    *,
    num_train_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    beta_schedule: str = "linear",
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a 1D beta schedule.

    returns:
        betas: [num_train_steps]
    """
    if num_train_steps <= 0:
        raise ValueError(f"num_train_steps must be positive, got {num_train_steps}.")
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError(f"beta_start and beta_end must be positive, got {beta_start}, {beta_end}.")
    if beta_start >= beta_end:
        raise ValueError(f"beta_start must be smaller than beta_end, got {beta_start} >= {beta_end}.")

    schedule_name = str(beta_schedule).lower()
    if schedule_name != "linear":
        raise ValueError(
            f"Unsupported beta_schedule '{beta_schedule}'. The minimal truncated diffusion utils "
            "currently only support 'linear'."
        )

    betas = torch.linspace(
        float(beta_start),
        float(beta_end),
        steps=int(num_train_steps),
        device=device,
        dtype=dtype,
    )  # [num_train_steps]
    if torch.any(betas >= 1.0):
        raise ValueError("All beta values must stay below 1.0.")
    return betas


def build_truncation_timesteps(
    *,
    num_train_steps: int,
    truncation_steps: int = 4,
    start_timestep: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Build a short descending timestep sequence for truncated diffusion.

    returns:
        truncation_timesteps: [truncation_steps] with values in [0, num_train_steps - 1]
    """
    if num_train_steps <= 0:
        raise ValueError(f"num_train_steps must be positive, got {num_train_steps}.")
    if truncation_steps <= 0:
        raise ValueError(f"truncation_steps must be positive, got {truncation_steps}.")

    max_timestep = int(num_train_steps - 1)
    if start_timestep is None:
        start_timestep = max_timestep
    start_timestep = int(start_timestep)

    if start_timestep < 0 or start_timestep > max_timestep:
        raise ValueError(
            f"start_timestep must be in [0, {max_timestep}], got {start_timestep}."
        )

    if truncation_steps == 1:
        return torch.tensor([start_timestep], device=device, dtype=torch.long)  # [1]

    raw = torch.linspace(
        float(start_timestep),
        0.0,
        steps=int(truncation_steps),
        device=device,
        dtype=torch.float32,
    )  # [truncation_steps]
    truncation_timesteps = torch.round(raw).long()
    truncation_timesteps = torch.unique_consecutive(truncation_timesteps, dim=0)

    if int(truncation_timesteps[0]) != start_timestep or int(truncation_timesteps[-1]) != 0:
        raise ValueError(
            "Failed to construct a valid descending truncation schedule. "
            f"Got {truncation_timesteps.tolist()} from start_timestep={start_timestep}."
        )
    if int(truncation_timesteps.numel()) != int(truncation_steps):
        raise ValueError(
            "Truncation timesteps collapsed after rounding. "
            f"Requested {truncation_steps} steps, got {truncation_timesteps.tolist()}. "
            "Use a larger start_timestep or a smaller truncation_steps value."
        )
    if not torch.all(truncation_timesteps[:-1] > truncation_timesteps[1:]):
        raise ValueError(
            f"truncation_timesteps must be strictly descending, got {truncation_timesteps.tolist()}."
        )
    return truncation_timesteps  # [truncation_steps]


def build_truncated_diffusion_schedule(
    *,
    num_train_steps: int = 16,
    truncation_steps: int = 4,
    start_timestep: int | None = None,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    beta_schedule: str = "linear",
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> TruncatedDiffusionSchedule:
    """Build the minimal schedule tensors needed by truncated diffusion."""
    betas = build_beta_schedule(
        num_train_steps=num_train_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        device=device,
        dtype=dtype,
    )  # [num_train_steps]
    alphas = 1.0 - betas  # [num_train_steps]
    alpha_bars = torch.cumprod(alphas, dim=0)  # [num_train_steps]
    alpha_bars_prev = torch.cat(
        [torch.ones(1, device=betas.device, dtype=betas.dtype), alpha_bars[:-1]],
        dim=0,
    )  # [num_train_steps]
    sqrt_alpha_bars = torch.sqrt(alpha_bars)  # [num_train_steps]
    sqrt_one_minus_alpha_bars = torch.sqrt((1.0 - alpha_bars).clamp(min=1e-12))  # [num_train_steps]
    posterior_variance = (
        betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp(min=1e-12)
    )  # [num_train_steps]
    truncation_timesteps = build_truncation_timesteps(
        num_train_steps=num_train_steps,
        truncation_steps=truncation_steps,
        start_timestep=start_timestep,
        device=betas.device,
    )  # [truncation_steps]

    schedule = TruncatedDiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
        posterior_variance=posterior_variance,
        truncation_timesteps=truncation_timesteps,
        num_train_steps=int(num_train_steps),
        truncation_steps=int(truncation_steps),
        beta_schedule=str(beta_schedule),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
    )
    validate_diffusion_schedule(schedule)
    return schedule


def validate_diffusion_schedule(schedule: TruncatedDiffusionSchedule) -> None:
    """Sanity-check a diffusion schedule."""
    expected_len = int(schedule.num_train_steps)
    one_d_tensors = {
        "betas": schedule.betas,
        "alphas": schedule.alphas,
        "alpha_bars": schedule.alpha_bars,
        "alpha_bars_prev": schedule.alpha_bars_prev,
        "sqrt_alpha_bars": schedule.sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": schedule.sqrt_one_minus_alpha_bars,
        "posterior_variance": schedule.posterior_variance,
    }
    for name, tensor in one_d_tensors.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 1:
            raise ValueError(f"schedule.{name} must be a 1D tensor, got {type(tensor)} with shape {getattr(tensor, 'shape', None)}.")
        if int(tensor.shape[0]) != expected_len:
            raise ValueError(
                f"schedule.{name} length {tensor.shape[0]} does not match num_train_steps {expected_len}."
            )

    truncation_timesteps = schedule.truncation_timesteps
    if not torch.is_tensor(truncation_timesteps) or truncation_timesteps.ndim != 1:
        raise ValueError(
            "schedule.truncation_timesteps must be a 1D tensor, "
            f"got {type(truncation_timesteps)} with shape {getattr(truncation_timesteps, 'shape', None)}."
        )
    if int(truncation_timesteps.shape[0]) != int(schedule.truncation_steps):
        raise ValueError(
            "schedule.truncation_timesteps length does not match truncation_steps: "
            f"{truncation_timesteps.shape[0]} != {schedule.truncation_steps}."
        )
    if torch.any(truncation_timesteps < 0) or torch.any(truncation_timesteps >= expected_len):
        raise ValueError(
            f"schedule.truncation_timesteps must stay in [0, {expected_len - 1}], got {truncation_timesteps.tolist()}."
        )
    if int(truncation_timesteps.numel()) > 1 and not torch.all(truncation_timesteps[:-1] > truncation_timesteps[1:]):
        raise ValueError(
            f"schedule.truncation_timesteps must be strictly descending, got {truncation_timesteps.tolist()}."
        )


def validate_action_chunk_tensor(
    action_chunks: torch.Tensor,
    *,
    name: str,
    action_chunk_dim: int | None = None,
) -> torch.Tensor:
    """Validate action chunks shaped as [D], [K, D], or [B, K, D]."""
    tensor = torch.as_tensor(action_chunks)
    if tensor.ndim not in {1, 2, 3}:
        raise ValueError(
            f"{name} must have shape [D], [K, D], or [B, K, D], got {tuple(tensor.shape)}."
        )
    if action_chunk_dim is not None and int(tensor.shape[-1]) != int(action_chunk_dim):
        raise ValueError(
            f"{name} last dim {tensor.shape[-1]} does not match action_chunk_dim {action_chunk_dim}."
        )
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor


def validate_noise_tensor(
    noise: torch.Tensor | None,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """Return Gaussian noise matching `like`.

    like: [D], [K, D], or [B, K, D]
    returns:
        noise: same shape as `like`
    """
    if noise is None:
        return torch.randn_like(like)

    noise_tensor = torch.as_tensor(noise, device=like.device, dtype=like.dtype)
    if tuple(noise_tensor.shape) != tuple(like.shape):
        raise ValueError(
            f"noise shape {tuple(noise_tensor.shape)} does not match target shape {tuple(like.shape)}."
        )
    return noise_tensor


def broadcast_timesteps(
    timesteps: int | torch.Tensor,
    *,
    target_shape: tuple[int, ...],
    num_train_steps: int,
    device: torch.device,
    allow_terminal_step: bool = False,
) -> torch.Tensor:
    """Broadcast timesteps to match action chunk leading dims.

    target_shape corresponds to:
        []
        [K]
        [B, K]

    returns:
        broadcasted_timesteps: same shape as target_shape
    """
    timestep_tensor = torch.as_tensor(timesteps, device=device, dtype=torch.long)
    min_timestep = -1 if allow_terminal_step else 0
    if torch.any(timestep_tensor < min_timestep) or torch.any(timestep_tensor >= int(num_train_steps)):
        raise ValueError(
            f"timesteps must stay in [{min_timestep}, {num_train_steps - 1}], got {timestep_tensor.detach().cpu().tolist()}."
        )

    if timestep_tensor.ndim > len(target_shape):
        raise ValueError(
            f"timesteps rank {timestep_tensor.ndim} cannot broadcast to target leading shape {target_shape}."
        )

    try:
        broadcasted = timestep_tensor + torch.zeros(target_shape, device=device, dtype=torch.long)
    except RuntimeError as exc:
        raise ValueError(
            f"timesteps shape {tuple(timestep_tensor.shape)} cannot broadcast to target leading shape {target_shape}."
        ) from exc
    return broadcasted


def extract_schedule_tensor(
    values: torch.Tensor,
    timesteps: int | torch.Tensor,
    *,
    target: torch.Tensor,
    num_train_steps: int,
    allow_terminal_step: bool = False,
    terminal_value: float = 1.0,
) -> torch.Tensor:
    """Gather 1D schedule values and expand them to action chunk shape.

    values: [num_train_steps]
    target:
        [D]
        [K, D]
        [B, K, D]

    returns:
        gathered: same leading dims as target, with trailing singleton dim
            [1]
            [K, 1]
            [B, K, 1]
    """
    if not torch.is_tensor(values) or values.ndim != 1:
        raise ValueError(f"values must be a 1D tensor, got {type(values)} with shape {getattr(values, 'shape', None)}.")
    if int(values.shape[0]) != int(num_train_steps):
        raise ValueError(
            f"values length {values.shape[0]} does not match num_train_steps {num_train_steps}."
        )

    target = validate_action_chunk_tensor(target, name="target")
    leading_shape = tuple(target.shape[:-1])
    expanded_timesteps = broadcast_timesteps(
        timesteps,
        target_shape=leading_shape,
        num_train_steps=num_train_steps,
        device=target.device,
        allow_terminal_step=allow_terminal_step,
    )

    if allow_terminal_step:
        terminal_mask = expanded_timesteps.eq(-1)
        safe_timesteps = expanded_timesteps.clamp(min=0)
    else:
        terminal_mask = None
        safe_timesteps = expanded_timesteps

    gathered = values.to(device=target.device, dtype=target.dtype).index_select(
        0,
        safe_timesteps.reshape(-1),
    ).reshape(leading_shape)

    if terminal_mask is not None:
        gathered = torch.where(
            terminal_mask,
            torch.full_like(gathered, float(terminal_value)),
            gathered,
        )

    while gathered.ndim < target.ndim:
        gathered = gathered.unsqueeze(-1)
    return gathered


def add_noise_to_action_chunks(
    action_chunks: torch.Tensor,
    *,
    timesteps: int | torch.Tensor,
    schedule: TruncatedDiffusionSchedule,
    noise: torch.Tensor | None = None,
    action_chunk_dim: int | None = None,
) -> torch.Tensor:
    """Apply forward diffusion noise to action chunks.

    action_chunks:
        [D]
        [K, D]
        [B, K, D]
    returns:
        noisy_action_chunks: same shape as action_chunks
    """
    validate_diffusion_schedule(schedule)
    clean = validate_action_chunk_tensor(
        action_chunks,
        name="action_chunks",
        action_chunk_dim=action_chunk_dim,
    )
    noise_tensor = validate_noise_tensor(noise, like=clean)

    sqrt_alpha_bar_t = extract_schedule_tensor(
        schedule.sqrt_alpha_bars,
        timesteps,
        target=clean,
        num_train_steps=schedule.num_train_steps,
    )  # [1], [K, 1], or [B, K, 1]
    sqrt_one_minus_alpha_bar_t = extract_schedule_tensor(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        target=clean,
        num_train_steps=schedule.num_train_steps,
    )  # [1], [K, 1], or [B, K, 1]

    noisy = sqrt_alpha_bar_t * clean + sqrt_one_minus_alpha_bar_t * noise_tensor
    if tuple(noisy.shape) != tuple(clean.shape):
        raise ValueError(
            f"noisy action chunk shape changed unexpectedly: {tuple(noisy.shape)} != {tuple(clean.shape)}."
        )
    return noisy


def add_noise_to_anchors(
    anchors: torch.Tensor,
    *,
    timesteps: int | torch.Tensor,
    schedule: TruncatedDiffusionSchedule,
    noise: torch.Tensor | None = None,
    action_chunk_dim: int | None = None,
) -> torch.Tensor:
    """Alias of add_noise_to_action_chunks with anchor-oriented naming.

    anchors:
        [K, D]
        [B, K, D]
    returns:
        noisy_anchors: same shape as anchors
    """
    return add_noise_to_action_chunks(
        anchors,
        timesteps=timesteps,
        schedule=schedule,
        noise=noise,
        action_chunk_dim=action_chunk_dim,
    )


def get_timestep_embedding(
    timesteps: int | torch.Tensor,
    embedding_dim: int,
    *,
    max_period: int = 10000,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build sinusoidal timestep embeddings.

    timesteps:
        []
        [B]
        [B, K]
    returns:
        timestep_embedding:
            [embedding_dim]
            [B, embedding_dim]
            [B, K, embedding_dim]
    """
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

    timestep_tensor = torch.as_tensor(timesteps, device=device, dtype=dtype)
    original_shape = tuple(timestep_tensor.shape)
    timestep_tensor = timestep_tensor.reshape(-1)  # [N]

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return torch.zeros(*original_shape, embedding_dim, device=timestep_tensor.device, dtype=dtype)

    exponent = -math.log(float(max_period)) * torch.arange(
        half_dim,
        device=timestep_tensor.device,
        dtype=dtype,
    ) / max(half_dim - 1, 1)
    freqs = torch.exp(exponent)  # [half_dim]
    args = timestep_tensor.unsqueeze(-1) * freqs.unsqueeze(0)  # [N, half_dim]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N, 2 * half_dim]

    if embedding_dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros(embedding.shape[0], 1, device=embedding.device, dtype=embedding.dtype)],
            dim=-1,
        )  # [N, embedding_dim]

    if len(original_shape) == 0:
        return embedding[0]  # [embedding_dim]
    return embedding.reshape(*original_shape, embedding_dim)


def predict_epsilon_from_x0(
    noisy_action_chunks: torch.Tensor,
    x0_pred: torch.Tensor,
    *,
    timesteps: int | torch.Tensor,
    schedule: TruncatedDiffusionSchedule,
    action_chunk_dim: int | None = None,
) -> torch.Tensor:
    """Recover epsilon implied by x_t and an x0 prediction.

    noisy_action_chunks:
        [D]
        [K, D]
        [B, K, D]
    x0_pred: same shape as noisy_action_chunks
    returns:
        epsilon_hat: same shape as noisy_action_chunks
    """
    validate_diffusion_schedule(schedule)
    x_t = validate_action_chunk_tensor(
        noisy_action_chunks,
        name="noisy_action_chunks",
        action_chunk_dim=action_chunk_dim,
    )
    x0_hat = validate_action_chunk_tensor(
        x0_pred,
        name="x0_pred",
        action_chunk_dim=action_chunk_dim,
    )
    if tuple(x_t.shape) != tuple(x0_hat.shape):
        raise ValueError(f"x_t shape {tuple(x_t.shape)} must match x0_pred shape {tuple(x0_hat.shape)}.")

    sqrt_alpha_bar_t = extract_schedule_tensor(
        schedule.sqrt_alpha_bars,
        timesteps,
        target=x_t,
        num_train_steps=schedule.num_train_steps,
    )
    sqrt_one_minus_alpha_bar_t = extract_schedule_tensor(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        target=x_t,
        num_train_steps=schedule.num_train_steps,
    ).clamp(min=1e-12)

    epsilon_hat = (x_t - sqrt_alpha_bar_t * x0_hat) / sqrt_one_minus_alpha_bar_t
    return epsilon_hat


def denoise_step_from_x0(
    noisy_action_chunks: torch.Tensor,
    x0_pred: torch.Tensor,
    *,
    timesteps: int | torch.Tensor,
    next_timesteps: int | torch.Tensor | None = None,
    schedule: TruncatedDiffusionSchedule,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
    action_chunk_dim: int | None = None,
) -> torch.Tensor:
    """One deterministic-or-nearly-deterministic reverse update from x0 prediction.

    This helper is intended for truncated diffusion inference.

    noisy_action_chunks:
        [D]
        [K, D]
        [B, K, D]
    x0_pred: same shape as noisy_action_chunks
    returns:
        next_action_chunks: same shape as noisy_action_chunks
    """
    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}.")

    validate_diffusion_schedule(schedule)
    x_t = validate_action_chunk_tensor(
        noisy_action_chunks,
        name="noisy_action_chunks",
        action_chunk_dim=action_chunk_dim,
    )
    x0_hat = validate_action_chunk_tensor(
        x0_pred,
        name="x0_pred",
        action_chunk_dim=action_chunk_dim,
    )
    if tuple(x_t.shape) != tuple(x0_hat.shape):
        raise ValueError(f"noisy_action_chunks shape {tuple(x_t.shape)} must match x0_pred shape {tuple(x0_hat.shape)}.")

    expanded_timesteps = broadcast_timesteps(
        timesteps,
        target_shape=tuple(x_t.shape[:-1]),
        num_train_steps=schedule.num_train_steps,
        device=x_t.device,
        allow_terminal_step=False,
    )
    if next_timesteps is None:
        next_timesteps = expanded_timesteps - 1

    alpha_bar_t = extract_schedule_tensor(
        schedule.alpha_bars,
        expanded_timesteps,
        target=x_t,
        num_train_steps=schedule.num_train_steps,
    )  # [1], [K, 1], or [B, K, 1]
    alpha_bar_next = extract_schedule_tensor(
        schedule.alpha_bars,
        next_timesteps,
        target=x_t,
        num_train_steps=schedule.num_train_steps,
        allow_terminal_step=True,
        terminal_value=1.0,
    )  # [1], [K, 1], or [B, K, 1]

    epsilon_hat = predict_epsilon_from_x0(
        x_t,
        x0_hat,
        timesteps=expanded_timesteps,
        schedule=schedule,
        action_chunk_dim=action_chunk_dim,
    )  # same shape as x_t

    sigma = torch.zeros_like(alpha_bar_t)
    if eta > 0.0:
        sigma = eta * torch.sqrt(
            ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t).clamp(min=1e-12))
            * (1.0 - alpha_bar_t / alpha_bar_next.clamp(min=1e-12)).clamp(min=0.0)
        )  # [1], [K, 1], or [B, K, 1]

    direction_coeff = torch.sqrt((1.0 - alpha_bar_next - sigma.square()).clamp(min=0.0))
    next_action_chunks = torch.sqrt(alpha_bar_next) * x0_hat + direction_coeff * epsilon_hat

    if eta > 0.0:
        noise_tensor = validate_noise_tensor(noise, like=x_t)
        next_action_chunks = next_action_chunks + sigma * noise_tensor

    if tuple(next_action_chunks.shape) != tuple(x_t.shape):
        raise ValueError(
            "denoise step changed the action chunk shape unexpectedly: "
            f"{tuple(next_action_chunks.shape)} != {tuple(x_t.shape)}."
        )
    return next_action_chunks
