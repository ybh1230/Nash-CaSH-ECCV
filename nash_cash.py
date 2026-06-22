from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NashCaSHConfig:
    enabled: bool = True
    floor: float = 0.65
    ceiling: float = 0.98
    temperature: float = 1.0
    full_prior: float = 1.25
    window_prior: float = 1.0
    eps: float = 1e-6


def _branch_payoff(
    output: torch.Tensor,
    query: torch.Tensor,
    prior: float,
    eps: float,
) -> torch.Tensor:
    output_float = output.detach().float()
    query_float = query.detach().float()

    compatibility = F.cosine_similarity(
        output_float.flatten(2, 3),
        query_float.flatten(2, 3),
        dim=-1,
    )
    compatibility = compatibility.mul(0.5).add(0.5).clamp_min(eps)

    energy = output_float.pow(2).mean(dim=(-1, -2)).sqrt()
    energy = energy / energy.mean(dim=1, keepdim=True).clamp_min(eps)
    energy = torch.sigmoid(energy - 1.0).clamp_min(eps)

    payoff = prior * (compatibility + energy + eps)
    return payoff.unsqueeze(-1).unsqueeze(-1)


def nash_equilibrium_mix(
    full_output: torch.Tensor,
    window_output: torch.Tensor,
    full_query: torch.Tensor,
    window_query: torch.Tensor,
    config: Optional[NashCaSHConfig] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if config is None:
        config = NashCaSHConfig()

    if not config.enabled or full_output.shape != window_output.shape:
        return full_output, None

    eps = max(float(config.eps), 1e-12)
    floor = min(max(float(config.floor), 0.0), 1.0)
    ceiling = min(max(float(config.ceiling), floor), 1.0)
    temperature = max(float(config.temperature), eps)

    full_payoff = _branch_payoff(full_output, full_query, float(config.full_prior), eps)
    window_payoff = _branch_payoff(window_output, window_query, float(config.window_prior), eps)

    full_payoff = full_payoff.pow(1.0 / temperature)
    window_payoff = window_payoff.pow(1.0 / temperature)

    # Nash social-welfare allocation:
    # argmax_a u_full log(a) + u_window log(1-a).
    nash_ratio = full_payoff / (full_payoff + window_payoff).clamp_min(eps)
    alpha = floor + (ceiling - floor) * nash_ratio
    alpha = alpha.to(dtype=full_output.dtype, device=full_output.device)

    mixed = alpha * full_output + (1.0 - alpha) * window_output
    return mixed, alpha
