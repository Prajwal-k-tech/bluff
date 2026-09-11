"""
Draw-Robust Metrics Suite (Tess / Buffy / Antigravity Specification).

Addresses Claim (c) & (e) in docs/claims-drawlock-calibration.md:
1. Bluff Calibration Error (BCE): Mean absolute calibration gap |p_hat - y_true|
2. Call Precision (A_call): Fraction of calls that correctly intercept bluffs
3. Lock-Point Hand Size (S_lock): Hand size at draw pile lock point (d = 0)
4. Shedding Efficiency at Lock: Difference in hand size (opp - bot) at lock point
5. Adaptation Velocity (t_half): Number of observations to reach 50% parameter convergence
6. Deception Elo: Margin-weighted Elo rating using post-lock hand differentials
"""

import math
from typing import List, Dict, Tuple, Optional


def compute_bluff_calibration_error(probs: List[float], labels: List[bool]) -> float:
    """Computes expected calibration error |p_hat - y_true|."""
    if not probs or len(probs) != len(labels):
        return 0.0
    return sum(abs(p - (1.0 if y else 0.0)) for p, y in zip(probs, labels)) / len(probs)


def compute_call_precision(correct_calls: int, total_calls: int) -> float:
    """Computes call accuracy A_call."""
    if total_calls == 0:
        return 0.0
    return (correct_calls / total_calls) * 100.0


def compute_shedding_differential(my_hand_lock: float, opp_hand_lock: float) -> float:
    """Positive value means bot shed more cards (smaller hand) than opponent at lock point."""
    return opp_hand_lock - my_hand_lock


def compute_adaptation_velocity(history: List[float], target: float, tolerance: float = 0.05) -> Optional[int]:
    """
    Returns number of turns until estimate stays within tolerance of target,
    or half-life t_half.
    """
    if not history:
        return None
    initial = history[0]
    half_target = initial + 0.5 * (target - initial)

    for turn, val in enumerate(history):
        if abs(val - target) <= abs(half_target - target):
            return turn
    return len(history)


def compute_deception_elo(ratings: Dict[str, float],
                          winner: Optional[str],
                          bot_a: str, bot_b: str,
                          s_lock_a: float, s_lock_b: float,
                          k_base: float = 32.0) -> Tuple[float, float]:
    """
    Computes updated Elo ratings weighted by shedding differential at lock point.
    Margin scaling: margin = 1.0 + 0.1 * (s_lock_loser - s_lock_winner).
    """
    ra = ratings.get(bot_a, 1500.0)
    rb = ratings.get(bot_b, 1500.0)

    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    eb = 1.0 - ea

    if winner == bot_a:
        sa, sb = 1.0, 0.0
        margin = max(0.5, 1.0 + 0.05 * (s_lock_b - s_lock_a))
    elif winner == bot_b:
        sa, sb = 0.0, 1.0
        margin = max(0.5, 1.0 + 0.05 * (s_lock_a - s_lock_b))
    else:
        sa, sb = 0.5, 0.5
        # Differential at draw rewards the player who had fewer cards at lock
        margin = max(0.5, 1.0 + 0.05 * abs(s_lock_b - s_lock_a))

    k = k_base * margin
    new_ra = ra + k * (sa - ea)
    new_rb = rb + k * (sb - eb)
    return new_ra, new_rb


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes Wilson 95% score confidence interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z ** 2) / (4.0 * (total ** 2))))
    return max(0.0, center - margin), min(1.0, center + margin)
