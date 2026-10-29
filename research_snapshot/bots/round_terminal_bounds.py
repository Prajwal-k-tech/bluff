"""Empirical completion bounds, not confidence intervals or policy guarantees."""
import math


def terminal_completion_bounds(wins, losses, unresolved):
    """Bound the sample mean if unresolved rollouts eventually score +/-1.

    Does not account for finite-sample uncertainty, model error, nontermination,
    adaptive sampling or distribution shift. No unresolved game is a draw.
    """
    if any(type(n) is not int or n < 0 for n in (wins, losses, unresolved)):
        raise ValueError("rollout counts must be nonnegative integers")
    total = wins + losses + unresolved
    if not total:
        return {"lower": -1.0, "upper": 1.0, "resolved_fraction": 0.0,
                "samples": 0}
    return {"lower": (wins - losses - unresolved) / total,
            "upper": (wins - losses + unresolved) / total,
            "resolved_fraction": (wins + losses) / total, "samples": total}


def fixed_sample_win_recommendation(scores, baseline_index=0, *, alpha=.05):
    """Conservative recommendation under a declared IID simulated-world model.

    Fixed-budget flat samples only, not adaptive UCT samples or optional stopping.
    Hoeffding bounds include unresolved samples as [-1,1] and use a union bound
    over all actions. Model misspecification is NOT covered. This is not a
    confidence guarantee for actual human opponents or a solved game.
    """
    if (not isinstance(scores, (list, tuple)) or not scores
            or type(baseline_index) is not int or not 0 <= baseline_index < len(scores)
            or type(alpha) not in (float, int) or not math.isfinite(alpha)
            or not 0 < alpha < 1):
        raise ValueError("invalid fixed-sample recommendation configuration")
    intervals = []
    samples = None
    for score in scores:
        b = terminal_completion_bounds(score["wins"], score["losses"], score["horizon_leaves"])
        if b["samples"] == 0 or (samples is not None and b["samples"] != samples):
            raise ValueError("equal positive fixed sample counts required")
        samples = b["samples"]
        radius = math.sqrt(2 * math.log(2 * len(scores) / alpha) / samples)
        intervals.append({"lower": max(-1., b["lower"] - radius),
                          "upper": min(1., b["upper"] + radius)})
    best = max(range(len(scores)), key=lambda i: intervals[i]["lower"])
    supported = (best != baseline_index and
                 intervals[best]["lower"] > intervals[baseline_index]["upper"])
    return {"selected_index": best if supported else baseline_index,
            "improvement_supported_under_model": supported,
            "intervals": intervals, "alpha": alpha, "samples": samples,
            "scope": "fixed IID simulated-world outcome means only; model error excluded"}
