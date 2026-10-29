"""Research-only actor-observation rollouts for sampled corrected-rule roots.

RootWorld proposals satisfy current constraints, not all past ownership/action
likelihoods. Older private opponent packets unknown to the root stay unknown;
we do NOT invent a consistent private history for that actor. This limits the
opponent model's historical memory, but never gives it the other actor's hand.
No actual live GameState is accepted by the root materializer.
"""

from copy import deepcopy
from dataclasses import dataclass
import random
import time
from math import isfinite

from bots.base import public_action_for_viewer
from bots.round_determinization import RootWorld, sample_root_world
from bots.round_observation import round_player_view
from cards import Card, Hand, Rank, Suit
from game_v2 import GameState


def materialize_root(hand, state, world: RootWorld):
    """Construct an isolated engine from own/redacted inputs and a proposal.

    Engine historical fields may retain redacted old cards; only current-round
    packets are reconstructed. Do not pass this engine to full-history metrics
    or represent its old unknown packets as privileged research ground truth.
    """
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1) or not isinstance(world, RootWorld):
        raise ValueError("invalid root viewer or proposal")
    hand = tuple(hand)
    cards = hand + world.opponent_hand + world.pile + world.draw_supply
    deck = {Card(rank, suit) for rank in Rank for suit in Suit}
    if len(cards) != 52 or set(cards) != deck:
        raise ValueError("sampled root must conserve all 52 distinct cards")
    expected = (len(hand), len(world.opponent_hand), len(world.pile), len(world.draw_supply))
    sizes = tuple(state.get(key) for key in
        ("hand_size", "opponent_hand_size", "pile_size", "draw_pile_size"))
    if expected != sizes:
        raise ValueError("proposal zone sizes disagree with observation")
    # GameState constructs a shuffled Deck even without deal(); keep simulations
    # from consuming live engine randomness merely by constructing a root.
    caller = random.getstate()
    try:
        game = GameState()
    finally:
        random.setstate(caller)
    game.hands = [Hand(hand), Hand(world.opponent_hand)] if viewer == 0 else [
        Hand(world.opponent_hand), Hand(hand)]
    game.pile, game.draw_pile = list(world.pile), list(world.draw_supply)
    game.current_player = viewer
    game.round_number = state["round_number"]
    game.turn_count = state["turn_number"]
    game.active_rank = state["active_rank"]
    game.actions = deepcopy(state["actions"])
    if state.get("private_pickups"):
        from bots.round_private_pickup_evidence import extract_private_pickup_evidence
        extract_private_pickup_evidence(state)
        # Preserve only the root observer's legally collected history. Never
        # invent the other actor's private pickups from sampled hidden cards.
        game.private_pickups = tuple(
            (viewer, tuple((packet["action_index"], tuple(packet["cards"]),
                            packet["claimed_rank"]) for packet in pickup))
            for pickup in state["private_pickups"])
    current = [a for a in game.actions if a.round_number == game.round_number]
    game.round_start_action_index = len(game.actions) - len(current)
    offset = 0
    for action in current:
        quantity = len(action.cards_played)
        packet = list(world.pile[offset:offset + quantity])
        if len(packet) != quantity or (action.player == viewer and packet != action.cards_played):
            raise ValueError("proposal contradicts current packet knowledge")
        action.cards_played = packet
        action.was_bluff = any(card.rank != action.claimed_rank for card in packet)
        offset += quantity
    if offset != len(world.pile):
        raise ValueError("current packets do not partition sampled pile")
    pending = state.get("last_action")
    if pending is not None:
        if not current or current[-1].action_index != pending.action_index:
            raise ValueError("pending claim disagrees with current history")
        game.pending_claim = current[-1]
        if tuple(game.pending_claim.cards_played) != world.pending_packet:
            raise ValueError("pending proposal packet disagrees with pile")
    elif world.pile or world.pending_packet or game.active_rank is not None:
        raise ValueError("opening root has an unresolved pile/rank")
    game.cards_played = [card for a in game.actions for card in a.cards_played
                         if isinstance(card, Card)]
    return game


def apply_choice(game, choice):
    """Apply a normal policy action, returning events for redacted notification."""
    player = game.current_player
    previous = game.latest_pending_claim
    kind = choice.get("action", choice.get("kind"))
    if kind == "play":
        ok, message = game.play_cards(player, choice.get("cards"), choice.get("rank"))
        observed = (previous, game.last_action)
    elif kind == "pass":
        ok, message = game.pass_turn(player)
        observed = (previous,)
    elif kind in ("call", "call_bluff", "challenge"):
        ok, message, resolved = game.call_bluff(player)
        observed = (resolved,)
    else:
        raise ValueError(f"unknown rollout action {kind!r}")
    if not ok:
        raise ValueError(f"rollout policy action rejected: {message}")
    return tuple(action for action in observed if action is not None)


@dataclass(frozen=True)
class RolloutResult:
    winner: int | None
    decisions: int
    termination: str
    leaf_card_advantage: float


def prepare_injected_forecast(policy, hand, view, choice):
    """Replace any selected-action forecast with one for an injected choice."""
    prepare = getattr(policy, "prepare_injected_root_forecast", None)
    if callable(prepare):
        prepare(hand, view, choice)
    elif getattr(policy, "register_injected_root_forecast", False):
        policy.policy_learner._pending.pop(len(view.get("actions", ())), None)
        policy._forecast_selected_play(hand, view, choice)


def rollout(game, policies, *, viewer, horizon, seed, root_choice=None,
            initial_engine_rng_state=None, decision_hook=None, checkpoint_hook=None,
            deadline=None, clock=time.monotonic, resumable_checkpoint=False,
            resume_pending=None,reuse_observed_views=False):
    """Consume isolated game and COPY policies; never mutate live profiles/RNG.

    Policies must be non-search continuation models. A horizon leaf is NOT a
    draw or estimated win probability; its card advantage is a bounded heuristic.
    Caller owns consistent root knowledge and the opponent continuation model.
    Optional checkpoint_hook receives isolated evaluator-only state at a
    nonterminal cutoff. Never expose its hidden game/RNG to an acting policy.
    For wall-cutoff reuse, opt into resumable_checkpoint and call resume_rollout:
    the checkpoint retains an already planned but unapplied action. Legacy
    horizon-only payloads do not promise wall-resume parity.
    """
    if (type(viewer) is not int or viewer not in (0, 1)
            or type(horizon) is not int or horizon < 1 or type(seed) is not int
            or len(policies) != 2 or policies[0] is policies[1]):
        raise ValueError("invalid rollout configuration")
    if deadline is not None and (type(deadline) not in (int, float) or not isfinite(deadline)):
        raise ValueError("deadline must be finite or None")
    if type(resumable_checkpoint) is not bool:
        raise ValueError('resumable checkpoint flag must be boolean')
    if type(reuse_observed_views) is not bool:
        raise ValueError('view reuse flag must be boolean')
    if resume_pending is not None and (
            not resumable_checkpoint or root_choice is not None
            or not isinstance(resume_pending,dict)
            or set(resume_pending)!={'choice','prepared'}
            or type(resume_pending['prepared']) is not bool
            or not isinstance(resume_pending['choice'],dict)):
        raise ValueError('invalid pending checkpoint choice')
    policies = deepcopy(policies)
    # Only exact audited continuation classes, never subclasses/search wrappers.
    from bots.round_pickup_adaptive_bot import RoundPickupAdaptiveBot
    from bots.round_risk_utility_actor import RoundRiskUtilityActor
    reusable=(RoundPickupAdaptiveBot,RoundRiskUtilityActor)
    observed_views={}
    for seat, policy in enumerate(policies):
        policy.player_id = seat
    caller = random.getstate()
    decisions = 0
    expired = False
    pending = deepcopy(resume_pending)
    if resumable_checkpoint and root_choice is not None:
        pending = {'choice':deepcopy(root_choice),'prepared':False}
    try:
        random.seed(seed)
        if initial_engine_rng_state is not None:
            random.setstate(initial_engine_rng_state)
        while not game.game_over and decisions < horizon:
            if deadline is not None and clock() >= deadline:
                expired = True
                break
            player = game.current_player
            if decisions==0 and resume_pending is not None:
                choice = deepcopy(resume_pending['choice'])
                prepare = not resume_pending['prepared']
            else:
                if decisions==0 and root_choice is not None:
                    choice=root_choice
                elif (reuse_observed_views and type(policies[player]) in reusable
                      and player in observed_views):
                    choice=policies[player]._choose_action_after_observe(
                        list(game.hands[player].cards),observed_views[player])
                else:
                    choice=policies[player].choose_action(list(game.hands[player].cards),
                                                         round_player_view(game,player))
                prepare = decisions==0 and root_choice is not None
            if prepare:
                view = round_player_view(game, player)
                prepare_injected_forecast(
                    policies[player], list(game.hands[player].cards), view, choice)
            if resumable_checkpoint:
                pending = {'choice':deepcopy(choice),'prepared':True}
            if decision_hook is not None:
                hook_rng = random.getstate()
                try:
                    decision_hook(game=deepcopy(game), policies=deepcopy(policies),
                                  choice=deepcopy(choice), decisions=decisions)
                finally:
                    random.setstate(hook_rng)
            if deadline is not None and clock() >= deadline:
                expired = True
                break
            observed = apply_choice(game, choice)
            pending = None
            decisions += 1
            for action in observed:
                for seat, policy in enumerate(policies):
                    policy.observe_action(public_action_for_viewer(action, seat),
                                          game.hands[1 - seat].size())
            # Match the arena's legal observation lifecycle, including pickup
            # facts delivered after resolution and before a terminal return.
            for seat, policy in enumerate(policies):
                observe_view = getattr(policy, "observe_player_view", None)
                if callable(observe_view):
                    view=round_player_view(game,seat)
                    observe_view(view)
                    if reuse_observed_views and type(policy) in reusable:
                        observed_views[seat]=view
        if not game.game_over and checkpoint_hook is not None:
            hook_rng = random.getstate()
            try:
                payload = dict(game=deepcopy(game), policies=deepcopy(policies),
                               engine_rng_state=hook_rng, decisions=decisions)
                if resumable_checkpoint:
                    payload.update(resumable_checkpoint=True,resume_pending=deepcopy(pending),viewer=viewer)
                checkpoint_hook(**payload)
            finally:
                random.setstate(hook_rng)
        return RolloutResult(game.winner if game.game_over else None, decisions,
            "win" if game.game_over else ("wall_deadline" if expired else "horizon"),
            (game.hands[1 - viewer].size() - game.hands[viewer].size()) / 52)
    finally:
        random.setstate(caller)


def resume_rollout(checkpoint, *, viewer, horizon, seed=0, deadline=None,
                   clock=time.monotonic, decision_hook=None, checkpoint_hook=None):
    """Resume ONLY explicit prepared-choice checkpoints; preserve caller state."""
    if (not isinstance(checkpoint,dict) or checkpoint.get('resumable_checkpoint') is not True
            or type(checkpoint.get('viewer')) is not int or checkpoint['viewer']!=viewer):
        raise ValueError('explicit resumable checkpoint/viewer required')
    return rollout(deepcopy(checkpoint['game']),checkpoint['policies'],viewer=viewer,
        horizon=horizon,seed=seed,initial_engine_rng_state=checkpoint['engine_rng_state'],
        resume_pending=checkpoint['resume_pending'],resumable_checkpoint=True,
        deadline=deadline,clock=clock,decision_hook=decision_hook,checkpoint_hook=checkpoint_hook)


def compare_root_choices(policy, opponent_model, hand, state, choices, *,
                         samples=4, horizon=8, seed=0, stratify_pending=False):
    """Paired flat root Monte Carlo; NOT an MCTS/SDMCTS implementation.

    Use the supplied policy's pre-reveal bluff forecast to choose latest-packet
    strata, not raw sampler frequencies. Explicitly fail on unsupported strata,
    invalid actions or budgets; never silently change the forecast or skip a
    losing branch. Candidate generation, opponent model and calibrated beliefs
    remain caller responsibilities. Leaf scoring is an experimental heuristic.
    """
    viewer = state.get("player_id")
    if (type(viewer) is not int or viewer not in (0, 1)
            or type(samples) is not int or samples < 1
            or type(horizon) is not int or horizon < 1 or type(seed) is not int
            or type(stratify_pending) is not bool
            or not isinstance(choices, (tuple, list)) or not choices
            or samples * horizon * len(choices) > 10000):
        raise ValueError("invalid root comparison or exceeds10000decision budget")
    focal = deepcopy(policy)
    focal.player_id = viewer
    pending = state.get("last_action")
    bluff_probability = None
    if pending is not None:
        bluff_probability, _ = focal._posterior_bluff(hand, state, pending)
        from math import isfinite
        if not isfinite(bluff_probability) or not 0 <= bluff_probability <= 1:
            raise ValueError("invalid root bluff forecast")
    strata = None
    if stratify_pending and pending is not None:
        if 0 < bluff_probability < 1:
            if samples < 2:
                raise ValueError("both nonzero truth strata require at least2samples")
            false_count = samples // 2
            true_count = samples - false_count
            # Integrate the supplied forecast, rather than drawing only one
            # label in a tiny Bernoulli sample. Within-stratum worlds remain
            # approximate; neither calibration nor strength is guaranteed.
            strata = ([(False, bluff_probability / false_count)] * false_count
                      + [(True, (1 - bluff_probability) / true_count)] * true_count)
        else:
            strata = [(bluff_probability == 0, 1 / samples)] * samples
    policies = [focal, opponent_model] if viewer == 0 else [opponent_model, focal]
    rng = random.Random(seed)
    totals = [{"wins": 0, "losses": 0, "horizon_leaves": 0, "value_sum": 0.0,
               "weighted_value_sum": 0.0}
              for _ in choices]
    for index in range(samples):
        if strata is not None:
            truth, weight = strata[index]
        else:
            truth = None if pending is None else rng.random() >= bluff_probability
            weight = 1 / samples
        world = sample_root_world(hand, state, rng, pending_truth=truth)
        branch_seed = rng.randrange(2 ** 63)
        for choice, total in zip(choices, totals):
            simulated = materialize_root(hand, state, world)
            result = rollout(simulated, policies, viewer=viewer, horizon=horizon,
                             seed=branch_seed, root_choice=choice)
            if result.termination == "win":
                won = result.winner == viewer
                total["wins" if won else "losses"] += 1
                value = 1.0 if won else -1.0
            else:
                total["horizon_leaves"] += 1
                value = result.leaf_card_advantage
            total["value_sum"] += value
            total["weighted_value_sum"] += weight * value
    return {"samples": samples, "horizon": horizon,
            "stratified_pending": strata is not None,
            "pending_bluff_forecast": bluff_probability,
            "scores": [{**total, "mean_heuristic_value": (
                total["weighted_value_sum"] if strata is not None
                else total["value_sum"] / samples)}
                       for total in totals]}
