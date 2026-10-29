"""Bounded corrected-rule runner; not a tournament or a strength claim.

Callers own independent opponent identities, policy RNG seeds and treatment
assignment. Reusing a policy preserves its behavioral profile; reset() clears
deal-local facts only. No database, historical weights or training is used.
"""

from dataclasses import dataclass, fields
import random
import math
import time
from copy import deepcopy

from bots.base import public_action_for_viewer
from bots.round_observation import round_player_view
from game_v2 import GameState, RULESET_ID


@dataclass(frozen=True)
class ClaimMetrics:
    """Evaluator-only counts. Zero denominators imply undefined rates, not 0%."""

    packets: int = 0
    bluff_packets: int = 0
    bluff_cards: int = 0
    voluntary_responses: int = 0
    voluntary_calls: int = 0
    correct_voluntary_calls: int = 0
    opponent_bluffs_responded_to: int = 0
    forced_calls: int = 0
    correct_forced_calls: int = 0
    bluffs_survived: int = 0
    bluff_cards_survived: int = 0
    bluffs_caught_voluntarily: int = 0
    bluffs_caught_forced: int = 0
    bluffs_pending: int = 0


def summarize_claims(actions) -> tuple[ClaimMetrics, ClaimMetrics]:
    """Count full engine-history packets, never redacted policy observations.

    Survival means the immediate challenge window closed via play or pass,
    not permanent shedding: a later challenge can return the entire pile.
    Pending packets are censored, not survivors. Automatic finals are separate.
    Do not supply these privileged counts to an opponent's online learner.
    """
    counts = [{field.name: 0 for field in fields(ClaimMetrics)} for _ in range(2)]
    for action in actions:
        player = action.player
        bluff = action.was_bluff
        quantity = len(action.cards_played)
        if (type(player) is not int or player not in (0, 1)
                or type(bluff) is not bool or not 1 <= quantity <= 4):
            raise ValueError("metrics require full, unredacted engine packets")
        own, opponent = counts[player], counts[1 - player]
        own["packets"] += 1
        if bluff:
            own["bluff_packets"] += 1
            own["bluff_cards"] += quantity
        response = getattr(action, "response_kind", None)
        forced = getattr(action, "response_was_forced", None)
        if response is None:
            if bluff:
                own["bluffs_pending"] += 1
            continue
        if (response not in {"play", "pass", "challenge"}
                or type(forced) is not bool
                or getattr(action, "response_player", None) != 1 - player
                or (forced and response != "challenge")):
            raise ValueError("inconsistent engine response metadata")
        if forced:
            opponent["forced_calls"] += 1
            if bluff:
                opponent["correct_forced_calls"] += 1
                own["bluffs_caught_forced"] += 1
        else:
            opponent["voluntary_responses"] += 1
            if bluff:
                opponent["opponent_bluffs_responded_to"] += 1
            if response == "challenge":
                opponent["voluntary_calls"] += 1
                if bluff:
                    opponent["correct_voluntary_calls"] += 1
                    own["bluffs_caught_voluntarily"] += 1
            elif bluff:
                own["bluffs_survived"] += 1
                own["bluff_cards_survived"] += quantity
    return tuple(ClaimMetrics(**seat) for seat in counts)


@dataclass(frozen=True)
class GameResult:
    ruleset_id: str
    seed: int
    winner: int | None
    decisions: int
    termination: str
    claim_metrics: tuple[ClaimMetrics, ClaimMetrics]


def play_game(policies, *, seed: int, max_decisions: int, game=None,
              starting_player: int = 0, decision_hook=None,
              deadline: float | None = None, pre_decision_hook=None) -> GameResult:
    """Run at most max_decisions, raising on policy errors (no hidden fallback).

    Optional game is an isolated fixture consumed in place. A normal deal is
    14/14/24; changing starting_player changes initiative, NOT card ownership.
    Seat-swapped experiments must also swap policies explicitly.
    Engine uses global random; this serial runner restores its caller's state.
    Do not run it concurrently in threads with other global-RNG consumers.
    deadline is an absolute monotonic timestamp, checked between decisions.
    It cannot interrupt a synchronous policy/hook; late choices are not applied.
    pre_decision_hook receives isolated evaluator-only game/policies/decisions
    BEFORE choose_action (no choice). decision_hook remains AFTER selection.
    """
    if len(policies) != 2 or policies[0] is policies[1]:
        raise ValueError("two distinct policy instances are required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(max_decisions) is not int or max_decisions <= 0:
        raise ValueError("max_decisions must be a positive integer")
    if type(starting_player) is not int or starting_player not in (0, 1):
        raise ValueError("starting_player must be 0 or 1")
    if game is not None and not isinstance(game, GameState):
        raise ValueError("fixture must use the corrected GameState")
    if deadline is not None and (type(deadline) not in (int, float)
                                 or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    caller_rng = random.getstate()
    try:
        random.seed(seed)
        if game is None:
            game = GameState()
            game.deal(14, random_start=False)
            game.current_player = starting_player
        for player, policy in enumerate(policies):
            policy.reset()
            policy.player_id = player
            for action in game.actions:
                policy.observe_action(public_action_for_viewer(action, player),
                                      game.hands[1 - player].size())
            # Full-history research policies need their initial private hand
            # even when the opponent moves first. Never recover it from turn1.
            observe_view = getattr(policy, "observe_player_view", None)
            if callable(observe_view):
                observe_view(round_player_view(game, player))

        decisions = 0
        deadline_reached = False
        while not game.game_over and decisions < max_decisions:
            if deadline is not None and time.monotonic() >= deadline:
                deadline_reached = True
                break
            player = game.current_player
            if pre_decision_hook is not None:
                hook_rng = random.getstate()
                try:
                    pre_decision_hook(game=deepcopy(game),policies=deepcopy(policies),
                                      decisions=decisions)
                finally:
                    random.setstate(hook_rng)
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_reached = True
                    break
            choice = policies[player].choose_action(
                list(game.hands[player].cards), round_player_view(game, player))
            if not isinstance(choice, dict):
                raise ValueError("policy returned a non-object decision")
            if decision_hook is not None:
                # Evaluator-only snapshots keep instrumentation away from live
                # game/policy state. Restore engine RNG even if the hook fails.
                hook_rng = random.getstate()
                try:
                    decision_hook(game=deepcopy(game), policies=deepcopy(policies),
                                  choice=deepcopy(choice),
                                  decisions=decisions)
                finally:
                    random.setstate(hook_rng)
            if deadline is not None and time.monotonic() >= deadline:
                deadline_reached = True
                break
            kind = choice.get("action", choice.get("kind"))
            previous = game.latest_pending_claim
            if kind == "play":
                ok, message = game.play_cards(player, choice.get("cards"), choice.get("rank"))
                observed = [previous, game.last_action]
            elif kind == "pass":
                ok, message = game.pass_turn(player)
                observed = [previous]
            elif kind in ("call", "call_bluff", "challenge"):
                ok, message, resolved = game.call_bluff(player)
                observed = [resolved]
            else:
                raise ValueError(f"unknown policy action: {kind!r}")
            if not ok:
                raise ValueError(f"player {player} policy action rejected: {message}")
            decisions += 1
            for action in observed:
                if action is not None:
                    for viewer, policy in enumerate(policies):
                        policy.observe_action(public_action_for_viewer(action, viewer),
                                              game.hands[1 - viewer].size())
            # Some information arrives on pile collection, not on the next
            # decision. Deliver it before terminal return/profile checkpoint.
            for viewer, policy in enumerate(policies):
                observe_view = getattr(policy, "observe_player_view", None)
                if callable(observe_view):
                    observe_view(round_player_view(game, viewer))
        if game.game_over and (type(game.winner) is not int or game.winner not in (0, 1)):
            raise ValueError("terminal engine state has no valid winner")
        return GameResult(RULESET_ID, seed, game.winner if game.game_over else None,
                          decisions, "win" if game.game_over else
                          "wall_deadline" if deadline_reached else "truncated",
                          summarize_claims(game.actions))
    finally:
        random.setstate(caller_rng)
