from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NashCaSHConfig:
    floor: float = 0.65
    ceiling: float = 0.98
    temperature: float = 1.0
    full_prior: float = 1.25
    window_prior: float = 1.0
    authority_momentum: float = 0.10
    authority_log_dir: Optional[str] = None
    authority_log_stride: int = 1
    authority_log_token_stride: int = 256
    authority_log_map_height: int = 48
    authority_log_map_width: int = 84
    authority_latent_frames: int = 0
    authority_latent_height: int = 0
    authority_latent_width: int = 0
    eps: float = 1e-6


def _branch_alignment(
    output: torch.Tensor,
    query: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output_float = output.detach().float()
    query_float = query.detach().float()

    cosine = F.cosine_similarity(output_float, query_float, dim=-1)
    # Negative query-response directions should not be treated as medium
    # reliability. They are treated as conflicting evidence for that token.
    alignment = cosine.clamp_min(0.0).clamp_min(eps)
    return alignment, output_float


def _branch_energy(output_float: torch.Tensor, eps: float) -> torch.Tensor:
    energy = output_float.pow(2).mean(dim=-1).sqrt()
    return energy.clamp_min(eps)


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.-]+", "_", name)[:160]


def log_authority_snapshot(
    authority: Optional[torch.Tensor],
    config: NashCaSHConfig,
    layer_name: str,
    step: Optional[int],
    call_index: int,
) -> None:
    if authority is None or not config.authority_log_dir:
        return

    stride = max(int(config.authority_log_stride), 1)
    if call_index % stride != 0:
        return

    with torch.no_grad():
        out_dir = Path(config.authority_log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        auth = authority.detach().float().cpu()
        if auth.ndim == 4 and auth.shape[-1] == 1:
            auth = auth.squeeze(-1)
        if auth.ndim != 3:
            return

        h = int(config.authority_latent_height)
        w = int(config.authority_latent_width)
        f = int(config.authority_latent_frames)
        expected_tokens = f * h * w if f > 0 and h > 0 and w > 0 else 0

        if expected_tokens > 0 and auth.shape[1] == expected_tokens:
            # Wan attention tensors are B x token x head.
            token_authority = auth.mean(dim=(0, 2))
        elif expected_tokens > 0 and auth.shape[2] == expected_tokens:
            # Fallback for B x head x token layouts.
            token_authority = auth.mean(dim=(0, 1))
        elif auth.shape[1] >= auth.shape[2]:
            token_authority = auth.mean(dim=(0, 2))
        else:
            token_authority = auth.mean(dim=(0, 1))

        token_stride = max(int(config.authority_log_token_stride), 1)
        token_trace = token_authority[::token_stride].contiguous()

        spatial_map = None
        if h > 0 and w > 0 and token_authority.numel() % (h * w) == 0:
            if f <= 0:
                f = token_authority.numel() // (h * w)
            if f * h * w == token_authority.numel():
                spatial_map = token_authority.reshape(f, h, w).mean(dim=0)
                map_h = max(int(config.authority_log_map_height), 1)
                map_w = max(int(config.authority_log_map_width), 1)
                spatial_map = F.adaptive_avg_pool2d(
                    spatial_map.unsqueeze(0).unsqueeze(0),
                    output_size=(min(map_h, h), min(map_w, w)),
                ).squeeze(0).squeeze(0).contiguous()

        record = {
            "layer": layer_name,
            "step": int(step) if step is not None else None,
            "call_index": int(call_index),
            "shape": tuple(authority.shape),
            "mean": float(token_authority.mean().item()),
            "std": float(token_authority.std(unbiased=False).item()),
            "min": float(token_authority.min().item()),
            "max": float(token_authority.max().item()),
            "token_stride": token_stride,
            "token_trace": token_trace,
            "spatial_map": spatial_map,
            "latent_frames": f,
            "latent_height": h,
            "latent_width": w,
        }

        step_name = "none" if step is None else f"{int(step):04d}"
        file_name = f"authority_call{call_index:05d}_step{step_name}_{_safe_name(layer_name)}.pt"
        torch.save(record, out_dir / file_name)


def nash_equilibrium_mix(
    full_output: torch.Tensor,
    window_output: torch.Tensor,
    full_query: torch.Tensor,
    window_query: torch.Tensor,
    config: Optional[NashCaSHConfig] = None,
    previous_authority: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if config is None:
        config = NashCaSHConfig()

    if full_output.shape != window_output.shape:
        return full_output, None

    eps = max(float(config.eps), 1e-12)
    floor = min(max(float(config.floor), 0.0), 1.0)
    ceiling = min(max(float(config.ceiling), floor), 1.0)
    temperature = max(float(config.temperature), eps)
    momentum = min(max(float(config.authority_momentum), 0.0), 0.95)

    full_alignment, full_float = _branch_alignment(full_output, full_query, eps)
    window_alignment, window_float = _branch_alignment(window_output, window_query, eps)

    full_energy = _branch_energy(full_float, eps)
    window_energy = _branch_energy(window_float, eps)
    energy_total = (full_energy + window_energy).clamp_min(eps)

    # Branch-relative confidence compares the two evidence responses at the
    # same token, avoiding foreground/background bias from global RMS means.
    full_confidence = (full_energy / energy_total).clamp_min(eps)
    window_confidence = (window_energy / energy_total).clamp_min(eps)

    full_reliability = float(config.full_prior) * (full_alignment + full_confidence + eps)
    window_reliability = float(config.window_prior) * (window_alignment + window_confidence + eps)
    full_reliability = full_reliability.unsqueeze(-1)
    window_reliability = window_reliability.unsqueeze(-1)

    full_reliability = full_reliability.pow(1.0 / temperature)
    window_reliability = window_reliability.pow(1.0 / temperature)

    # Disagreement-aware Nash bargaining:
    # argmax_a u_full log(a-d_full) + u_window log(1-a-d_window).
    # floor is d_full and 1-ceiling is d_window, so the bounded authority is
    # the Nash solution over the surplus rather than an ad-hoc post-clamp.
    nash_ratio = full_reliability / (full_reliability + window_reliability).clamp_min(eps)
    alpha = floor + (ceiling - floor) * nash_ratio

    if (
        momentum > 0.0
        and previous_authority is not None
        and previous_authority.shape == alpha.shape
    ):
        previous = previous_authority.to(device=alpha.device, dtype=alpha.dtype)
        alpha = momentum * previous + (1.0 - momentum) * alpha

    alpha = alpha.to(dtype=full_output.dtype, device=full_output.device)

    mixed = alpha * full_output + (1.0 - alpha) * window_output
    return mixed, alpha
