from __future__ import annotations

import torch
from einops import rearrange


def ensure_context_shape(z_context: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize latent context to [B, T_ctx, latent_dim].

    z_context:
        [B, latent_dim] or [B, T_ctx, latent_dim]

    returns:
        normalized_context: [B, T_ctx, latent_dim]
        squeezed_context: True when the input was [B, latent_dim]
    """
    if not torch.is_tensor(z_context):
        raise TypeError(f"z_context must be a torch.Tensor, got {type(z_context)}.")
    if z_context.ndim == 2:
        return z_context.unsqueeze(1), True  # [B, 1, latent_dim]
    if z_context.ndim == 3:
        return z_context, False  # [B, T_ctx, latent_dim]
    raise ValueError(
        "z_context must have shape [B, latent_dim] or [B, T_ctx, latent_dim], "
        f"got {tuple(z_context.shape)}."
    )


def maybe_expand_candidate_dim(action_blocks: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize action blocks to [B, S, R, block_action_dim].

    action_blocks:
        [B, R, block_action_dim] or [B, S, R, block_action_dim]

    returns:
        normalized_action_blocks: [B, S, R, block_action_dim]
        squeezed_candidates: True when the input was [B, R, block_action_dim]
    """
    if not torch.is_tensor(action_blocks):
        raise TypeError(f"action_blocks must be a torch.Tensor, got {type(action_blocks)}.")
    if action_blocks.ndim == 3:
        return action_blocks.unsqueeze(1), True  # [B, 1, R, block_action_dim]
    if action_blocks.ndim == 4:
        return action_blocks, False  # [B, S, R, block_action_dim]
    raise ValueError(
        "action_blocks must have shape [B, R, block_action_dim] or "
        f"[B, S, R, block_action_dim], got {tuple(action_blocks.shape)}."
    )


def freeze_module_parameters(module: torch.nn.Module) -> list[bool]:
    """Freeze module parameters and return their previous requires_grad flags."""
    previous_flags = [parameter.requires_grad for parameter in module.parameters()]
    module.requires_grad_(False)
    return previous_flags


def restore_module_requires_grad(module: torch.nn.Module, previous_flags: list[bool]) -> None:
    """Restore requires_grad flags previously returned by freeze_module_parameters(...)."""
    parameters = list(module.parameters())
    if len(parameters) != len(previous_flags):
        raise ValueError(
            "Cannot restore requires_grad flags because the module parameter count changed: "
            f"{len(parameters)} != {len(previous_flags)}."
        )
    for parameter, requires_grad in zip(parameters, previous_flags, strict=True):
        parameter.requires_grad_(requires_grad)


def latent_rollout(
    *,
    world_model: torch.nn.Module,
    z_context: torch.Tensor,
    action_blocks: torch.Tensor,
    history_size: int = 3,
    return_sequence: bool = False,
    freeze_world_model: bool = True,
) -> dict[str, torch.Tensor]:
    """Roll out latent dynamics from an existing latent context.

    This helper mirrors the predictor loop inside JEPA.rollout(...), but it
    starts from latent embeddings instead of requiring info["pixels"].

    Inputs:
        z_context:
            [B, latent_dim] or [B, T_ctx, latent_dim]
        action_blocks:
            [B, R, block_action_dim] or [B, S, R, block_action_dim]

    Outputs:
        z_terminal:
            [B, S, latent_dim] or [B, latent_dim] when action_blocks was [B, R, block_action_dim]
        z_traj, optional:
            [B, S, T_ctx + R, latent_dim] or [B, T_ctx + R, latent_dim]

    Notes:
        - world_model parameters can be frozen, but the function does not use
          torch.no_grad(); gradients can still flow from z_terminal to
          action_blocks through world_model.action_encoder and world_model.predict.
        - action_blocks must already use the same prepared action space expected
          by world_model.action_encoder, e.g. [action_block * action_dim].
        - When z_context has T_ctx > 1, the helper prepends zero action
          context of shape [B, S, T_ctx - 1, block_action_dim]. This keeps
          predictor input lengths aligned without requiring pixels/history.
    """
    if history_size <= 0:
        raise ValueError(f"history_size must be positive, got {history_size}.")
    if not hasattr(world_model, "action_encoder"):
        raise TypeError("world_model must expose an action_encoder module.")
    if not hasattr(world_model, "predict"):
        raise TypeError("world_model must expose a predict(emb, act_emb) method.")

    z_context, _ = ensure_context_shape(z_context)  # [B, T_ctx, latent_dim]
    action_blocks, squeezed_candidates = maybe_expand_candidate_dim(action_blocks)  # [B, S, R, block_action_dim]

    if z_context.shape[0] != action_blocks.shape[0]:
        raise ValueError(
            "Batch size mismatch between z_context and action_blocks: "
            f"{z_context.shape[0]} != {action_blocks.shape[0]}."
        )
    if z_context.shape[1] <= 0:
        raise ValueError("z_context must contain at least one latent step.")
    if action_blocks.shape[1] <= 0:
        raise ValueError("action_blocks must contain at least one candidate.")
    if action_blocks.shape[2] <= 0:
        raise ValueError("action_blocks must contain at least one rollout block.")
    if action_blocks.shape[3] <= 0:
        raise ValueError("action_blocks must have a positive block_action_dim.")

    batch_size, num_candidates, rollout_blocks = action_blocks.shape[:3]

    action_blocks = action_blocks.to(device=z_context.device, dtype=z_context.dtype)
    z_expanded = z_context.unsqueeze(1).expand(
        batch_size,
        num_candidates,
        int(z_context.shape[1]),
        int(z_context.shape[2]),
    )  # [B, S, T_ctx, latent_dim]

    previous_flags: list[bool] | None = None
    if freeze_world_model:
        previous_flags = freeze_module_parameters(world_model)

    try:
        emb = rearrange(z_expanded, "b s t d -> (b s) t d").clone()  # [B*S, T_ctx, latent_dim]
        act = rearrange(action_blocks, "b s r a -> (b s) r a")  # [B*S, R, block_action_dim]
        context_action_steps = max(0, int(z_context.shape[1]) - 1)
        if context_action_steps > 0:
            zero_context_actions = torch.zeros(
                int(act.shape[0]),
                context_action_steps,
                int(act.shape[-1]),
                dtype=act.dtype,
                device=act.device,
            )  # [B*S, T_ctx - 1, block_action_dim]
            act = torch.cat([zero_context_actions, act], dim=1)  # [B*S, T_ctx - 1 + R, block_action_dim]

        for step in range(rollout_blocks):
            current_action = act[:, : context_action_steps + step + 1, :]  # [B*S, T_ctx + step, block_action_dim]
            act_emb = world_model.action_encoder(current_action)
            emb_trunc = emb[:, -history_size:, :]  # [B*S, min(history_size, T), latent_dim]
            act_trunc = act_emb[:, -history_size:, :]  # [B*S, min(history_size, T), action_emb_dim]
            pred_emb = world_model.predict(emb_trunc, act_trunc)[:, -1:, :]  # [B*S, 1, latent_dim]
            emb = torch.cat([emb, pred_emb], dim=1)  # [B*S, T_ctx + step + 1, latent_dim]

        z_traj = rearrange(
            emb,
            "(b s) t d -> b s t d",
            b=batch_size,
            s=num_candidates,
        )  # [B, S, T_ctx + R, latent_dim]
        z_terminal = z_traj[:, :, -1, :]  # [B, S, latent_dim]
    finally:
        if previous_flags is not None:
            restore_module_requires_grad(world_model, previous_flags)

    if squeezed_candidates:
        z_terminal = z_terminal[:, 0, :]  # [B, latent_dim]
        if return_sequence:
            z_traj = z_traj[:, 0, :, :]  # [B, T_ctx + R, latent_dim]

    output: dict[str, torch.Tensor] = {"z_terminal": z_terminal}
    if return_sequence:
        output["z_traj"] = z_traj
    return output


__all__ = [
    "ensure_context_shape",
    "maybe_expand_candidate_dim",
    "latent_rollout",
]
