from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ODEStep:
    flow_t: float
    dt: float


@dataclass(frozen=True)
class AnimationFramePlan:
    frame_index: int
    frame_t: float
    ode_steps: tuple[ODEStep, ...]


def build_animation_plan(num_frames: int, ode_steps: int) -> tuple[AnimationFramePlan, ...]:
    """Create a generation plan that solves a complete flow ODE for every animation frame."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if ode_steps <= 0:
        raise ValueError("ode_steps must be positive")

    dt = 1.0 / float(ode_steps)
    flow_steps = tuple(ODEStep(flow_t=i * dt, dt=dt) for i in range(ode_steps))
    denom = max(num_frames - 1, 1)
    return tuple(
        AnimationFramePlan(frame_index=i, frame_t=i / float(denom), ode_steps=flow_steps)
        for i in range(num_frames)
    )


def restore_normalized_vertices(
    vertices: Sequence[Sequence[float]],
    centroid: Sequence[float],
    scale: float,
) -> list[list[float]]:
    """Invert datasetv2.normalize_geometry: restored = normalized / scale + centroid."""
    if abs(float(scale)) < 1e-12:
        raise ValueError("scale must be non-zero")
    c = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
    return [
        [
            float(vertex[0]) / float(scale) + c[0],
            float(vertex[1]) / float(scale) + c[1],
            float(vertex[2]) / float(scale) + c[2],
        ]
        for vertex in vertices
    ]


def checkpoint_has_frame_condition(state_dict: Mapping[str, object] | Iterable[str]) -> bool:
    keys = state_dict.keys() if hasattr(state_dict, "keys") else state_dict
    return any(str(key).startswith("frame_mlp.") for key in keys)


def checkpoint_has_vertex_identity(state_dict: Mapping[str, object] | Iterable[str]) -> bool:
    keys = state_dict.keys() if hasattr(state_dict, "keys") else state_dict
    return any(str(key) == "vertex_embed" for key in keys)


def checkpoint_supports_sequence_mesh(state_dict: Mapping[str, object] | Iterable[str]) -> bool:
    keys = tuple(state_dict.keys() if hasattr(state_dict, "keys") else state_dict)
    return checkpoint_has_frame_condition(keys) and checkpoint_has_vertex_identity(keys)
