"""Research-only likelihoods for first responses to an opening claim.

The sampler and policies receive only the focal hand and its redacted public
view. Sampled worlds and physical actions are consumed internally; the public
result contains categorical response counts/probabilities only.
"""

from collections import Counter
from copy import deepcopy
from math import isfinite
import random

from bots.base import public_action_for_viewer
from bots.round_determinization import sample_root_world
from bots.round_observation import round_player_view
from bots.round_policy import RoundHonestBot, RoundMathBot, RoundRandomBot
from bots.round_rollout import apply_choice, materialize_root
from cards import Card, Rank


EXPERTS = ("Math", "Honest", "Random", "unknown")
FAMILIES = ("play", "pass", "challenge")
ALPHABETS = {
    "family": FAMILIES,
    "family_quantity": ("play1", "play2", "play3", "play4", "pass", "challenge"),
}
SAMPLES = 16
RESTART_SHARE = 0.05
JEFFREYS = 0.5


def _validate_root(hand, state):
    if not isinstance(hand, (list, tuple)) or any(not isinstance(c, Card) for c in hand):
        raise TypeError("hand must be a sequence of physical cards")
    if not isinstance(state, dict):
        raise TypeError("state must be a redacted round_player_view")
    forbidden = {"game", "hands", "opponent_hand", "target", "target_class", "truth", "was_bluff"}
    if forbidden.intersection(state):
        raise ValueError("state contains non-public or evaluation-only fields")
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1) or state.get("viewer") != viewer:
        raise ValueError("invalid focal seat")
    if state.get("own_hand") != list(hand):
        raise ValueError("hand must match the focal player's public-view hand")
    sizes = tuple(state.get(k) for k in ("hand_size", "opponent_hand_size", "pile_size", "draw_pile_size"))
    if (any(type(n) is not int or n < 0 for n in sizes) or sum(sizes) != 52
            or sizes[0] != len(hand) or len(set(hand)) != len(hand)):
        raise ValueError("invalid public card counts or focal hand")
    actions, pending = state.get("actions"), state.get("last_action")
    round_no = state.get("round_number")
    if not isinstance(actions, (list, tuple)) or len(actions) != 1 or pending is None:
        raise ValueError("only a one-claim opening response root is supported")
    if (type(round_no) is not int or round_no != 1 or state.get("turn_number") != 1
            or state.get("private_pickups", []) != []
            or getattr(actions[0], "action_index", None) != getattr(pending, "action_index", None)
            or getattr(pending, "action_index", None) != 0
            or getattr(pending, "round_number", None) != round_no
            or getattr(pending, "player", None) != viewer
            or getattr(pending, "claimed_rank", None) != state.get("active_rank")
            or not isinstance(state.get("active_rank"), Rank)
            or getattr(pending, "bluff_called", None) is not False
            or getattr(pending, "response_kind", None) is not None):
        raise ValueError("root must contain one unresolved opponent opening claim")
    qty = len(getattr(pending, "cards_played", ()))
    if (not 1 <= qty <= 4 or sizes[2] != qty
            or state.get("own_pile_cards") != list(getattr(pending, "cards_played", ()))
            or state.get("can_call_bluff") is not False):
        raise ValueError("opening claim is resolved or malformed")
    return viewer, qty


def _symbol(choice):
    action = choice.get("action", choice.get("kind")) if isinstance(choice, dict) else None
    if action == "play":
        cards = choice.get("cards")
        qty = len(cards) if isinstance(cards, (list, tuple)) else 0
        if type(qty) is not int or not 1 <= qty <= 4:
            raise ValueError("expert returned an invalid play quantity")
        return "play", f"play{qty}"
    if action in ("pass",):
        return "pass", "pass"
    if action in ("call_bluff", "call", "challenge"):
        return "challenge", "challenge"
    raise ValueError("expert returned an unsupported action")


def _expert(name, seed):
    return {"Math": RoundMathBot, "Honest": RoundHonestBot,
            "Random": RoundRandomBot}[name](seed=seed)


def response_likelihoods(hand, state, rng, *, samples=SAMPLES, max_states=10000):
    """Estimate each baseline's first-response law over exactly 16 sampled worlds."""
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be random.Random")
    if type(samples) is not int or samples != SAMPLES:
        raise ValueError("the frozen diagnostic requires exactly 16 worlds")
    if type(max_states) is not int or not 1 <= max_states <= 10000:
        raise ValueError("max_states must be an integer from 1 through 10000")
    viewer, _ = _validate_root(hand, state)
    before = random.getstate()
    counts = {name: {alphabet: Counter() for alphabet in ALPHABETS}
              for name in ("Math", "Honest", "Random")}
    pending = state["last_action"]
    played = tuple(state["own_pile_cards"])
    pre_hand = tuple(hand) + played
    prestate = dict(state)
    prestate.update({
        "hand_size": len(pre_hand), "pile_size": 0,
        "active_rank": None, "actions": [], "last_action": None,
        "own_hand": list(pre_hand), "own_pile_cards": [],
        "turn_number": 0, "can_play": True,
        "can_call_bluff": False, "can_pass": False,
    })
    try:
        for _ in range(samples):
            # The corrected engine uses module-global randomness for actions
            # such as recycling on pass. Seed that isolated engine RNG from the
            # caller-owned local RNG, then restore the ambient process state.
            ambient = random.getstate()
            random.seed(rng.randrange(2**32))
            try:
                world = sample_root_world(pre_hand, prestate, rng, pending_truth=None,
                                          max_states=max_states)
                base = materialize_root(pre_hand, prestate, world)
                ok, message = base.play_cards(viewer, list(played), pending.claimed_rank)
                if not ok:
                    raise ValueError(f"opening claim rejected in sampled world: {message}")
                actor = 1 - viewer
                for name in ("Math", "Honest", "Random"):
                    game = deepcopy(base)
                    policy = _expert(name, rng.randrange(2**32))
                    policy.reset()
                    policy.player_id = actor
                    game.current_player = actor
                    policy.observe_action(
                        public_action_for_viewer(game.latest_pending_claim, actor),
                        game.hands[viewer].size())
                    actor_hand = list(game.hands[actor].cards)
                    observation = round_player_view(game, actor)
                    choice = policy.choose_action(actor_hand, observation)
                    family_symbol, quantity_symbol = _symbol(choice)
                    public_events = apply_choice(game, choice)
                    # An engine-accepted transition is the legality oracle. Mirror
                    # the policy's actor-view observation delivery.
                    for event in public_events:
                        policy.observe_action(public_action_for_viewer(event, actor),
                                              game.hands[1 - actor].size())
                    if callable(getattr(policy, "observe_player_view", None)):
                        policy.observe_player_view(round_player_view(game, actor))
                    counts[name]["family"][family_symbol] += 1
                    counts[name]["family_quantity"][quantity_symbol] += 1
            finally:
                random.setstate(ambient)
        if random.getstate() != before:
            raise RuntimeError("response simulation mutated global RNG state")
    finally:
        random.setstate(before)

    likelihoods = {"unknown": {}}
    for name in ("Math", "Honest", "Random"):
        likelihoods[name] = {}
        for alphabet, categories in ALPHABETS.items():
            denom = samples + JEFFREYS * len(categories)
            likelihoods[name][alphabet] = {
                category: (counts[name][alphabet][category] + JEFFREYS) / denom
                for category in categories
            }
    for alphabet, categories in ALPHABETS.items():
        likelihoods["unknown"][alphabet] = {c: 1 / len(categories) for c in categories}
    return {"likelihoods": likelihoods, "samples": samples,
            "alphabets": {key: tuple(value) for key, value in ALPHABETS.items()},
            "experts": EXPERTS, "method": "16 sampled corrected-rule public first responses; Jeffreys smoothing"}


def update_posterior(posterior, likelihoods, alphabet, observation, *, restart_share=RESTART_SHARE):
    """Apply uniform fixed-share restart, then a categorical Bayes update."""
    if alphabet not in ALPHABETS or observation not in ALPHABETS[alphabet]:
        raise ValueError("unsupported alphabet or observation")
    if (not isinstance(posterior, dict) or set(posterior) != set(EXPERTS)
            or not isinstance(likelihoods, dict) or set(likelihoods) != set(EXPERTS)):
        raise ValueError("posterior and likelihoods must contain the four canonical experts")
    if (isinstance(restart_share, bool) or not isinstance(restart_share, (int, float))
            or not isfinite(restart_share) or not 0 <= restart_share <= 1):
        raise ValueError("restart_share must be finite and in [0, 1]")
    total = sum(posterior.values())
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0
            for v in posterior.values()) or not isfinite(total) or total <= 0):
        raise ValueError("posterior must be finite nonnegative mass")
    prior = {expert: ((1 - restart_share) * posterior[expert] / total
                      + restart_share / len(EXPERTS)) for expert in EXPERTS}
    masses = {}
    for expert in EXPERTS:
        dist = likelihoods[expert].get(alphabet) if isinstance(likelihoods[expert], dict) else None
        if not isinstance(dist, dict) or set(dist) != set(ALPHABETS[alphabet]):
            raise ValueError("malformed expert categorical likelihood")
        vals = list(dist.values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0 for v in vals):
            raise ValueError("likelihoods must be finite nonnegative mass")
        norm = sum(vals)
        if not isfinite(norm) or norm <= 0:
            raise ValueError("expert likelihood must have positive mass")
        masses[expert] = prior[expert] * dist[observation] / norm
    evidence = sum(masses.values())
    if not isfinite(evidence) or evidence <= 0:
        raise ValueError("observation has zero predictive probability")
    return {expert: masses[expert] / evidence for expert in EXPERTS}


def uniform_posterior():
    return {expert: 1 / len(EXPERTS) for expert in EXPERTS}
