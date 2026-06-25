from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(v_pred: torch.Tensor, x_1: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v_pred, x_1 - x_0)


def temporal_velocity_loss(v_seq: torch.Tensor) -> torch.Tensor:
    if v_seq.ndim < 4 or v_seq.shape[1] < 2:
        return v_seq.new_tensor(0.0)
    return F.mse_loss(v_seq[:, 1:], v_seq[:, :-1])


def composite_stdf_loss(
    *,
    v_pred: torch.Tensor,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    config,
    id_loss: torch.Tensor | None = None,
    sem_loss: torch.Tensor | None = None,
    temporal_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    flow = flow_matching_loss(v_pred, x_1, x_0)
    identity = id_loss if id_loss is not None else flow.new_tensor(0.0)
    semantic = sem_loss if sem_loss is not None else flow.new_tensor(0.0)
    temporal = temporal_loss if temporal_loss is not None else flow.new_tensor(0.0)
    total = (
        config.LAMBDA_FLOW * flow
        + config.LAMBDA_ID * identity
        + config.LAMBDA_SEM * semantic
        + config.LAMBDA_TEMP * temporal
    )
    logs = {
        "loss_total": float(total.detach().cpu()),
        "loss_flow": float(flow.detach().cpu()),
        "loss_id": float(identity.detach().cpu()),
        "loss_sem": float(semantic.detach().cpu()),
        "loss_temp": float(temporal.detach().cpu()),
    }
    return total, logs
