"""Research joint opponent-hypothesis/pending-truth root proposal.

Conditional truth probabilities are caller-supplied model assumptions. This
coupling enforces compatibility; it does not calibrate those assumptions.
"""
from dataclasses import dataclass
import math
import random

from bots.round_determinization import RootWorld, sample_root_world
from bots.round_policy_mixture_actor import RoundPolicyMixtureActor

HYPOTHESES = ("Honest", "Math", "Random", "unknown")


@dataclass(frozen=True)
class JointPolicyRoot:
    world: RootWorld
    hypothesis: str
    pending_truth: bool | None


def sample_joint_policy_root(hand, view, rng, *, weights, truth_probabilities=None):
    """Draw H then truth conditional on H, then the compatible physical world.

    Unsupported nonzero components raise. Silently dropping them would alter
    the supplied distribution and hide a broken belief model.
    """
    if not isinstance(rng, random.Random):
        raise TypeError("caller-owned random.Random required")
    # Reuse the frozen actor's strict normalized-weight contract.
    RoundPolicyMixtureActor(weights)
    pending = view.get("last_action")
    if pending is None:
        if truth_probabilities is not None:
            raise ValueError("opening roots have no pending truth variable")
    else:
        if (not isinstance(truth_probabilities, dict)
                or set(truth_probabilities) != set(HYPOTHESES)
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or not 0 <= v <= 1 for v in truth_probabilities.values())
                or truth_probabilities["Honest"] != 1.):
            raise ValueError("pending root requires compatible per-hypothesis truth probabilities")
    hypothesis = rng.choices(HYPOTHESES, [weights[h] for h in HYPOTHESES])[0]
    truth = None if pending is None else rng.random() < truth_probabilities[hypothesis]
    world = sample_root_world(hand, view, rng, pending_truth=truth,
                              opponent_pile_truthful=hypothesis == "Honest")
    return JointPolicyRoot(world, hypothesis, truth)


def actor_for_joint_root(root, *, seed=None):
    if not isinstance(root, JointPolicyRoot):
        raise TypeError("joint root proposal required")
    return RoundPolicyMixtureActor({h: float(h == root.hypothesis) for h in HYPOTHESES}, seed)
