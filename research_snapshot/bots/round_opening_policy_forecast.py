"""Opening-only action-likelihood forecast under actual declared baselines."""
from collections import Counter
import random
from bots.round_opening_forecast import _opening_inputs
from bots.round_determinization import sample_root_world
from bots.round_rollout import materialize_root
from bots.round_observation import round_player_view
from bots.round_policy import RoundHonestBot, RoundMathBot
from bots.round_opening_likelihood import opening_claim_distribution
from bots.round_opening_exact import exact_opening_style_stats
from cards import Rank, Card
from bots.round_policy_mixture_actor import RoundPolicyMixtureActor


def forecast_opening_policies(hand, view, *, seed=0, samples=128):
    """Sample pre-claim hands, then condition on the observed rank/quantity.

    Genuine baseline policies choose using their own legal views. This is not
    the toy mixed-uniform Math surrogate. Zero sampled match means insufficient
    resolution, not impossible observation. Restrict to first-deal opening to
    avoid pretending to reconstruct unknown private opponent history.
    """
    if type(seed) is not int or type(samples) is not int or not 1 <= samples <= 128:
        raise ValueError("invalid opening-policy sampling budget")
    if not isinstance(view, dict) or set(view).intersection({"hands", "opponent_hand", "game", "truth", "target"}):
        raise ValueError("legal focal view required")
    rank, quantity, before = _opening_inputs(hand, view)
    if len(view["actions"]) != 1 or view["round_number"] != 1 or view.get("private_pickups"):
        raise ValueError("only first-deal opening supported")
    rng = random.Random(seed)
    counts = {name: Counter() for name in ("Honest", "Math", "Random")}
    quantity_counts = {name: Counter() for name in counts}
    for _ in range(samples):
        world = sample_root_world(hand, before, rng)
        game = materialize_root(hand, before, world)
        actor = 1-view["player_id"]
        game.current_player = actor
        actor_hand = list(game.hands[actor].cards)
        actor_view = round_player_view(game, actor)
        for name, constructor in (("Honest", RoundHonestBot), ("Math", RoundMathBot)):
            policy = constructor(rng.randrange(2**63))
            choice = policy.choose_action(actor_hand, actor_view)
            true = all(c.rank == choice["rank"] for c in choice["cards"])
            if len(choice["cards"]) == quantity:
                quantity_counts[name]["true" if true else "false"] += 1
                if choice["rank"] == rank:
                    counts[name]["true" if true else "false"] += 1
        distribution = opening_claim_distribution(Counter(c.rank for c in actor_hand), "random_uniform")
        for (r, q, true), probability in distribution.items():
            if q == quantity:
                quantity_counts["Random"]["true" if true else "false"] += probability
                if r == rank:
                    counts["Random"]["true" if true else "false"] += probability
    def summarize(counter):
        total = counter["true"] + counter["false"]
        return {"likelihood": total/samples, "conditional_truth": counter["true"]/total if total else None,
                "true_mass": counter["true"]/samples, "false_mass": counter["false"]/samples}
    return {"samples": samples, "rank_quantity": {h: summarize(c) for h, c in counts.items()},
        "quantity_only": {h: summarize(c) for h, c in quantity_counts.items()},
        "scope": "first-opening sampled action likelihood, not full history posterior; zero matches unresolved"}


def exact_first_deal_policy_forecast(hand, view):
    """Exact default14-card baseline claim laws under a uniform unseen deal.

    Math: a pair exists, truthful pair EV2.7 exceeds best bluff EV2.6;
    therefore largest truthful group wins. Sorted hand tie picks lowest rank.
    Honest picks highest tied rank. This equivalence is NOT assumed outside
    this first-deal14-card/default-parameter scope or for human opponents.
    """
    if not isinstance(view, dict) or set(view).intersection({"hands", "opponent_hand", "game", "truth", "target"}):
        raise ValueError("legal focal view required")
    if (not isinstance(hand, (list, tuple)) or any(not isinstance(c, Card) for c in hand)
            or len(set(hand)) != len(hand) or view.get("own_hand") != list(hand)):
        raise ValueError("focal hand must match legal observation")
    rank, q, before = _opening_inputs(hand, view)
    if (len(view["actions"]) != 1 or view["round_number"] != 1 or view.get("private_pickups")
            or len(hand) != 14 or before["opponent_hand_size"] != 14 or before["draw_pile_size"] != 24):
        raise ValueError("exact actual-policy law requires standard first deal")
    counts = Counter(c.rank for c in hand)
    available = {r: 4-counts[r] for r in Rank}
    low = exact_opening_style_stats(available, 14, rank, q, honest_tie_rule="lowest")
    high = exact_opening_style_stats(available, 14, rank, q, honest_tie_rule="highest")
    selected = {"Math": low["honest_max"], "Honest": high["honest_max"],
                "Random": low["random_uniform"]}
    return {"rank_quantity": {h: {"likelihood": row["likelihood"],
        "true_mass": row["joint_truth"], "false_mass": row["joint_bluff"],
        "conditional_truth": None if row["conditional_bluff"] is None else 1-row["conditional_bluff"]}
        for h, row in selected.items()}, "scope": "exact first-deal default-policy laws under uniform unseen deal; not human/history posterior"}


def exact_first_deal_mixture(hand, view, weights):
    """Condition supplied policy weights on observed claim, without mutating them.

    Unknown uses the explicit Random law, not a universal model of humans.
    The supplied weights may be heuristic; exact likelihoods do not make them
    calibrated. No identity-specific exception or silent zero-support fallback.
    """
    RoundPolicyMixtureActor(weights)
    forecast = exact_first_deal_policy_forecast(hand, view)["rank_quantity"]
    laws = {**forecast, "unknown": forecast["Random"]}
    evidence = sum(weights[h]*laws[h]["likelihood"] for h in weights)
    if evidence <= 0:
        raise ValueError("observed opening has zero likelihood under declared mixture")
    posterior = {h: weights[h]*laws[h]["likelihood"]/evidence for h in weights}
    truth = sum(weights[h]*laws[h]["true_mass"] for h in weights)/evidence
    return {"weights": posterior, "probability_truth": truth,
        "truth_probabilities": {h: laws[h]["conditional_truth"] for h in weights},
        "claim_likelihood": evidence,
        "scope": "first-deal exact conditional declared-policy mixture; unknown=Random; prior calibration not established"}
