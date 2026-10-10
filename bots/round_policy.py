"""Small information-safe policies for fixed-rank Bluff rounds.

These policies consume the corrected round API rather than the historical
free-rank bot API.  Their math policy is a bounded card-advantage heuristic,
not a solved or globally optimal strategy.
"""

from __future__ import annotations

from collections import Counter
from math import comb
from random import Random
from typing import Any, Iterable, Optional

from cards import Card, Rank
from game import Action
from game_v2 import RULESET_ID
from bots.round_card_memory import opponent_rank_bounds


_PROFILE_VERSION = 1
_BLUFF_PRIOR = 0.30
_CALL_PRIOR = 0.35
_PRIOR_STRENGTH = 4.0
_RANKS = tuple(Rank)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _rank(value: Any) -> Optional[Rank]:
    if value is None:
        return None
    try:
        return value if isinstance(value, Rank) else Rank(value)
    except (TypeError, ValueError):
        return None


def _packet_size(action: Any) -> int:
    cards = _field(action, "cards_played")
    if cards is not None:
        try:
            return len(cards)
        except TypeError:
            pass
    quantity = _field(action, "quantity", 0)
    return quantity if isinstance(quantity, int) and quantity >= 0 else 0


def _latest_action(state: dict) -> Any:
    latest = state.get("last_action")
    if latest is not None:
        return latest
    actions = state.get("actions") or []
    return actions[-1] if actions else None


def _unique_known_cards(hand: Iterable[Card], state: dict) -> list[Card]:
    """Cards whose identities this viewer knows and which are out of supply."""
    cards = list(hand) + list(state.get("own_pile_cards") or [])
    return list(dict.fromkeys(card for card in cards if isinstance(card, Card)))


def _known_rank_count(rank: Rank, hand: Iterable[Card], state: dict) -> int:
    return sum(card.rank == rank for card in _unique_known_cards(hand, state))


def _hypergeom_at_least(population: int, successes: int, sample: int,
                        threshold: int) -> float:
    """P(X >= threshold) for a hypergeometric sample, using exact combs."""
    if threshold <= 0:
        return 1.0
    if (population < 0 or successes < 0 or successes > population
            or sample < 0 or sample > population):
        return 0.0
    denominator = comb(population, sample)
    if denominator == 0:
        return 0.0
    low = max(threshold, 0, sample - (population - successes))
    high = min(successes, sample)
    if low > high:
        return 0.0
    numerator = sum(comb(successes, count)
                    * comb(population - successes, sample - count)
                    for count in range(low, high + 1))
    return numerator / denominator


def _unseen_rank_pool(rank: Rank, hand: Iterable[Card], state: dict) -> tuple[int, int]:
    known = _unique_known_cards(hand, state)
    population = 52 - len(known)
    rank_copies = 4 - sum(card.rank == rank for card in known)
    if population < 0 or rank_copies < 0:
        return max(0, population), 0
    return population, rank_copies


def _can_prove_false(hand: Iterable[Card], state: dict, action: Any) -> bool:
    claim_rank = _rank(_field(action, "claimed_rank"))
    quantity = _packet_size(action)
    if claim_rank is None or not 1 <= quantity <= 4:
        return False
    if (_known_rank_count(claim_rank, hand, state) + quantity) > 4:
        return True
    viewer = state.get("player_id", state.get("viewer"))
    remaining = state.get("opponent_hand_size")
    if (type(viewer) is not int or viewer not in (0, 1)
            or type(remaining) is not int or remaining < 0):
        return False
    _, before = opponent_rank_bounds(state.get("actions"), viewer, action)
    # Guaranteed other-rank cards occupied slots BEFORE this hidden packet.
    # Use cardinality only: a heuristic near-zero probability is not proof.
    non_claim = sum(count for rank, count in before.items() if rank != claim_rank)
    return quantity > remaining + quantity - non_claim


def _context(qty: int, pile_size: int) -> str:
    qty_band = "1" if qty == 1 else "2" if qty == 2 else "3-4"
    pile_band = "1-2" if pile_size <= 2 else "3-5" if pile_size <= 5 else "6+"
    return f"{qty_band}|{pile_band}"


def _valid_bool(value: Any) -> bool:
    return isinstance(value, bool)


class _RoundPolicy:
    """Shared legality, telemetry, and profile boundary for round bots."""

    policy_name = "round_policy"

    def __init__(self, seed: Optional[int] = None):
        self.rng = Random(seed)
        self.player_id: Optional[int] = None
        self.last_decision_telemetry: dict[str, Any] = {}
        # These are deal-local observation identities only; no card labels or
        # action history are persisted in a behavioral profile.
        self._resolved_action_objects: set[Any] = set()

    def reset(self) -> None:
        """Clear deal-local state while preserving learned behavior."""
        self._resolved_action_objects.clear()
        self.last_decision_telemetry = {}

    def reset_profile(self) -> None:
        """Clear only cross-game behavioral evidence (stateless by default)."""

    def _set_viewer(self, state: dict) -> None:
        viewer = state.get("player_id", state.get("viewer"))
        if viewer is not None:
            self.player_id = viewer

    def _legal_families(self, hand: list[Card], state: dict) -> list[str]:
        families = []
        if state.get("can_play", bool(hand)) and hand:
            families.append("play")
        last = _latest_action(state)
        if (state.get("can_call_bluff", False) and last is not None
                and (self.player_id is None
                     or _field(last, "player") != self.player_id)):
            families.append("call_bluff")
        if state.get("can_pass", False):
            families.append("pass")
        if not families:
            raise ValueError("round state has no legal action for this policy")
        return families

    @staticmethod
    def _play(cards: list[Card], rank: Rank) -> dict:
        return {"action": "play", "cards": cards, "rank": rank}

    @staticmethod
    def _simple(action: str) -> dict:
        return {"action": action}

    def choose_action(self, hand: list[Card], state: dict) -> dict:
        raise NotImplementedError

    def observe_action(self, action: Action, opponent_hand_size: int) -> None:
        """Compatibility hook; stateless policies intentionally ignore it."""

    def to_dict(self) -> dict:
        return {"version": _PROFILE_VERSION, "policy": self.policy_name,
                "ruleset_id": RULESET_ID}

    def restore_opponent_profile(self, data: dict) -> None:
        if (not isinstance(data, dict) or type(data.get("version")) is not int
                or data.get("version") != _PROFILE_VERSION
                or data.get("policy") != self.policy_name
                or data.get("ruleset_id") != RULESET_ID):
            raise ValueError("unsupported round-policy behavioral profile")


class RoundRandomBot(_RoundPolicy):
    """Uniform legal action-family, then quantity and physical subset."""

    policy_name = "round_random"

    def choose_action(self, hand: list[Card], state: dict) -> dict:
        self._set_viewer(state)
        families = self._legal_families(hand, state)
        family = self.rng.choice(families)
        if family == "call_bluff":
            result = self._simple("call_bluff")
        elif family == "pass":
            result = self._simple("pass")
        else:
            active = _rank(state.get("active_rank"))
            claimed_rank = active if active is not None else self.rng.choice(_RANKS)
            quantity = self.rng.randint(1, min(4, len(hand)))
            result = self._play(self.rng.sample(hand, quantity), claimed_rank)
        self.last_decision_telemetry = {
            "policy": self.policy_name, "action": result["action"],
            "legal_families": tuple(families),
        }
        return result


class RoundHonestBot(_RoundPolicy):
    """Truthful packets, with a certified call and a pass if stuck."""

    policy_name = "round_honest"

    @staticmethod
    def _truthful_packet(hand: list[Card], rank: Rank) -> list[Card]:
        return [card for card in hand if card.rank == rank][:4]

    def _honest_play(self, hand: list[Card], state: dict) -> Optional[dict]:
        if not hand:
            return None
        active = _rank(state.get("active_rank"))
        counts = Counter(card.rank for card in hand)
        # A one-packet truthful finish wins even under automatic final review.
        if len(hand) <= 4 and len(counts) == 1:
            finish_rank = next(iter(counts))
            if active is None or active == finish_rank:
                return self._play(list(hand), finish_rank)
        if active is None:
            # Largest held group; a rank value breaks ties reproducibly.
            rank = max(counts, key=lambda item: (counts[item], int(item)))
        else:
            rank = active
        cards = self._truthful_packet(hand, rank)
        return self._play(cards, rank) if cards else None

    def choose_action(self, hand: list[Card], state: dict) -> dict:
        self._set_viewer(state)
        families = self._legal_families(hand, state)
        last = _latest_action(state)
        if ("call_bluff" in families and last is not None
                and _can_prove_false(hand, state, last)):
            result = self._simple("call_bluff")
            reason = "certified_contradiction"
        else:
            play = self._honest_play(hand, state) if "play" in families else None
            if play is not None:
                result, reason = play, "truthful_packet"
            elif "pass" in families:
                result, reason = self._simple("pass"), "no_truthful_packet"
            elif "play" in families and hand:
                # Defensive custom-state fallback: no pass, no proof, and no
                # truthful packet. This is a forced lie only if the engine
                # presents no legal honest option; ordinary recycle rules
                # leave passing available in this position.
                active = _rank(state.get("active_rank"))
                rank = active if active is not None else hand[0].rank
                result, reason = self._play([hand[0]], rank), "forced_only_play"
            elif "call_bluff" in families:
                # This can occur only in a malformed custom state where the
                # challenge is the sole legal action; never add a voluntary call.
                result, reason = self._simple("call_bluff"), "sole_legal_action"
            else:
                raise ValueError("no legal honest action")
        self.last_decision_telemetry = {
            "policy": self.policy_name, "action": result["action"],
            "reason": reason, "legal_families": tuple(families),
        }
        return result


class _RoundMathBase(_RoundPolicy):
    """Shared bounded card-advantage planner for Math and Adaptive."""

    bluff_prior = _BLUFF_PRIOR
    call_prior = _CALL_PRIOR

    def _bluff_rate(self, quantity: int, pile_size: int) -> float:
        return self.bluff_prior

    def _call_rate(self, quantity: int, pile_size: int) -> float:
        return self.call_prior

    def _hold_probability(self, hand, state, rank, sample_size, threshold,
                          *, before_pending=False):
        bounds_pair = state.get("_opponent_rank_bounds")
        if bounds_pair is None:
            bounds_pair = opponent_rank_bounds(
                state.get("actions"), self.player_id, state.get("last_action"))
        bounds = bounds_pair[1 if before_pending else 0]
        known_total = sum(bounds.values())
        known_rank = bounds.get(rank, 0)
        population, rank_copies = _unseen_rank_pool(rank, hand, state)
        # Known minimum copies are reserved in the opponent's hand. Model the
        # remaining slots as an exchangeable sample; this is a heuristic belief,
        # not exact Bayesian inference under an opponent's selective actions.
        return _hypergeom_at_least(
            population - known_total, rank_copies - known_rank,
            sample_size - known_total, threshold - known_rank)

    def _posterior_bluff(self, hand: list[Card], state: dict,
                         action: Any) -> tuple[float, float]:
        rank = _rank(_field(action, "claimed_rank"))
        quantity = _packet_size(action)
        if rank is None or not 1 <= quantity <= 4:
            return 1.0, 0.0
        # The latest packet was part of the opponent's hand immediately before
        # it was played. Unresolved packet identities stay unknown.
        before_hand = max(0, int(state.get("opponent_hand_size", 0)) + quantity)
        can_truthfully_hold = self._hold_probability(
            hand, state, rank, before_hand, quantity, before_pending=True)
        prior = min(1.0, max(0.0, self._bluff_rate(
            quantity, int(state.get("pile_size", quantity)))))
        denom = prior + (1.0 - prior) * can_truthfully_hold
        posterior = 1.0 if denom <= 0 or can_truthfully_hold <= 0 else prior / denom
        return posterior, can_truthfully_hold

    def _certifiable_probability(self, hand: list[Card], state: dict,
                                 rank: Rank, quantity: int) -> float:
        threshold = 5 - quantity  # opponent can prove false with > 4-q copies
        opponent_size = max(0, int(state.get("opponent_hand_size", 0)))
        # The opponent also knows the identities of its own packets still in
        # the center. Two-player pile size minus our own packets gives their
        # public quantity, without inspecting any of their hidden identities.
        opponent_center = max(0, int(state.get("pile_size", 0))
                              - len(state.get("own_pile_cards") or []))
        return self._hold_probability(
            hand, state, rank, opponent_size + opponent_center, threshold)

    @staticmethod
    def _shape_tiebreak(cards: tuple[Card, ...], hand_counts: Counter) -> tuple[int, int]:
        # Prefer disposing of singleton ranks; avoid breaking a held group when
        # utilities are tied. This only resolves ties, not the action value.
        singletons = sum(hand_counts[card.rank] == 1 for card in cards)
        split_group_cards = sum(hand_counts[card.rank] > 1 for card in cards)
        return singletons, -split_group_cards

    def _play_candidates(self, hand: list[Card], state: dict):
        """Exact representatives for THIS scorer, not a general search mask.

        Utility depends only on rank, quantity and truthfulness. Within each
        class maximize singleton shedding, then preserve the original physical
        combination order. Future packet-dependent lookahead must expand this.
        """
        active = _rank(state.get("active_rank"))
        ranks = (active,) if active is not None else _RANKS
        counts = Counter(card.rank for card in hand)
        preference = sorted(range(len(hand)),
                            key=lambda i: (counts[hand[i].rank] != 1, i))
        for quantity in range(1, min(4, len(hand)) + 1):
            candidates = []
            for rank in ranks:
                matching = tuple(i for i, card in enumerate(hand) if card.rank == rank)
                if len(matching) >= quantity:
                    candidates.append((matching[:quantity], rank, True))
                bluff = preference[:quantity]
                if all(hand[i].rank == rank for i in bluff):
                    replacement = next((i for i in preference if hand[i].rank != rank), None)
                    if replacement is None:
                        continue
                    # All selected cards share the claimed rank; replace the
                    # latest one to retain earliest-combination tie-breaking.
                    bluff[-1] = replacement
                candidates.append((tuple(sorted(bluff)), rank, False))
            for indices, rank, truthful in sorted(
                    candidates, key=lambda item: (item[0], int(item[1]))):
                yield tuple(hand[i] for i in indices), rank, truthful

    def _plan(self, hand: list[Card], state: dict) -> tuple[dict, dict]:
        state = dict(state)
        state["_opponent_rank_bounds"] = opponent_rank_bounds(
            state.get("actions"), self.player_id, state.get("last_action"))
        families = self._legal_families(hand, state)
        last = _latest_action(state)
        # Exact truthful finish is a guaranteed win under automatic final review.
        if "play" in families and hand and len(hand) <= 4:
            counts = Counter(card.rank for card in hand)
            if len(counts) == 1:
                finish_rank = next(iter(counts))
                active = _rank(state.get("active_rank"))
                if active is None or finish_rank == active:
                    action = self._play(list(hand), finish_rank)
                    return action, {"reason": "truthful_terminal_finish", "utility": None}

        options: list[tuple[float, int, tuple[int, int], dict, dict]] = []
        best_play = None
        call_estimates = {}
        candidates_seen = 0
        hand_counts = Counter(card.rank for card in hand)
        pile_before = max(0, int(state.get("pile_size", 0)))
        if "play" in families:
            for packet, claimed_rank, truthful in self._play_candidates(hand, state):
                quantity = len(packet)
                pile_after = pile_before + quantity
                estimate_key = (claimed_rank, quantity)
                if estimate_key not in call_estimates:
                    call_estimates[estimate_key] = (
                        min(1.0, max(0.0, self._call_rate(quantity, pile_after))),
                        self._certifiable_probability(
                            hand, state, claimed_rank, quantity),
                    )
                prior_call, certifiable = call_estimates[estimate_key]
                call_probability = max(prior_call, certifiable)
                if quantity == len(hand):
                    # Final packets are challenged by the rules, not by an
                    # opponent's voluntary call policy.
                    call_probability = 1.0
                utility = (quantity + call_probability * pile_after if truthful
                           else quantity - call_probability * pile_after)
                result = self._play(list(packet), claimed_rank)
                detail = {
                    "truthful": truthful, "quantity": quantity,
                    "claimed_rank": claimed_rank, "utility": utility,
                    "call_probability": call_probability,
                    "certifiable_probability": certifiable,
                }
                # Truthful action is the deterministic tie preference.
                candidate = (utility, int(truthful),
                             self._shape_tiebreak(packet, hand_counts),
                             result, detail)
                if best_play is None or candidate[:3] > best_play[:3]:
                    best_play = candidate
                candidates_seen += 1
        if best_play is not None:
            options.append(best_play)
        if "call_bluff" in families and last is not None:
            posterior, can_truth = self._posterior_bluff(hand, state, last)
            pile = max(0, int(state.get("pile_size", 0)))
            call_value = (2.0 * posterior - 1.0) * pile
            options.append((call_value, 0, (0, 0), self._simple("call_bluff"), {
                "reason": "challenge_ev", "posterior_bluff": posterior,
                "probability_can_truthfully_hold": can_truth,
                "utility": call_value,
            }))
        if "pass" in families:
            options.append((-1.0, 0, (0, 0), self._simple("pass"), {
                "reason": "pass_cost", "utility": -1.0,
            }))
        if not options:
            raise ValueError("round state produced no candidate actions")
        selected = max(options, key=lambda item: (item[0], item[1], item[2]))
        result, detail = selected[3], selected[4]
        detail = dict(detail)
        detail["scored_candidate_count"] = candidates_seen
        detail["candidate_count"] = (
            sum(comb(len(hand), q) for q in range(1, min(4, len(hand)) + 1))
            * (1 if _rank(state.get("active_rank")) is not None else len(_RANKS))
            if "play" in families else 0)
        detail["legal_families"] = tuple(families)
        detail["opponent_known_rank_lower_bounds"] = {
            rank.display(): count
            for rank, count in state["_opponent_rank_bounds"][0].items()
            if count
        }
        return result, detail

    def choose_action(self, hand: list[Card], state: dict) -> dict:
        self._set_viewer(state)
        action, detail = self._plan(hand, state)
        self.last_decision_telemetry = {"policy": self.policy_name,
                                        "action": action["action"], **detail}
        return action


class RoundMathBot(_RoundMathBase):
    """Fixed-prior, transparent short-horizon utility baseline."""

    policy_name = "round_math"


class RoundAdaptiveBot(_RoundMathBase):
    """Math backbone with smoothed public call/revealed-bluff profiles."""

    policy_name = "round_adaptive"

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._call_evidence: dict[str, list[int]] = {}
        self._bluff_evidence: dict[str, list[int]] = {}

    @staticmethod
    def _counts(evidence: dict[str, list[int]]) -> tuple[int, int]:
        successes = sum(value[0] for value in evidence.values())
        failures = sum(value[1] for value in evidence.values())
        return successes, failures

    @staticmethod
    def _smoothed_rate(prior: float, evidence: dict[str, list[int]], key: str) -> float:
        successes, failures = RoundAdaptiveBot._counts(evidence)
        global_mean = (_PRIOR_STRENGTH * prior + successes) / (
            _PRIOR_STRENGTH + successes + failures)
        local = evidence.get(key, [0, 0])
        return (_PRIOR_STRENGTH * global_mean + local[0]) / (
            _PRIOR_STRENGTH + local[0] + local[1])

    def _bluff_rate(self, quantity: int, pile_size: int) -> float:
        return self._smoothed_rate(_BLUFF_PRIOR, self._bluff_evidence,
                                  _context(quantity, pile_size))

    def _call_rate(self, quantity: int, pile_size: int) -> float:
        return self._smoothed_rate(_CALL_PRIOR, self._call_evidence,
                                  _context(quantity, pile_size))

    def reset_profile(self) -> None:
        self._call_evidence.clear()
        self._bluff_evidence.clear()

    @staticmethod
    def _add(evidence: dict[str, list[int]], key: str, success: bool) -> None:
        counts = evidence.setdefault(key, [0, 0])
        counts[0 if success else 1] += 1

    def observe_action(self, action: Action, opponent_hand_size: int) -> None:
        if self.player_id is None:
            return
        # The adapter supplies detached copies. Engine-issued deal-local IDs
        # deduplicate those too; retain references for legacy test fixtures.
        action_index = _field(action, "action_index")
        observation_key = (("action", action_index)
                           if type(action_index) is int and action_index >= 0
                           else action)
        try:
            if observation_key in self._resolved_action_objects:
                return
        except TypeError:
            return
        player = _field(action, "player")
        quantity = _packet_size(action)
        if not 1 <= quantity <= 4:
            return
        forced = _field(action, "response_was_forced")
        if forced is not False:
            return
        pile = max(0, int(_field(action, "pile_size_before", 0))) + quantity
        key = _context(quantity, pile)

        if player == self.player_id:
            response_call = _field(action, "response_was_call")
            response_player = _field(action, "response_player")
            if (_valid_bool(response_call)
                    and (response_player is None or response_player != self.player_id)):
                # False includes either a voluntary play continuation or pass.
                self._add(self._call_evidence, key, response_call)
                self._resolved_action_objects.add(observation_key)
            return

        # Opponent truth is evidence only after a legal reveal. Never inspect
        # `was_bluff` for an uncalled opponent play; its value is hidden.
        if _field(action, "bluff_called") is True:
            revealed_bluff = _field(action, "was_bluff")
            if _valid_bool(revealed_bluff):
                self._add(self._bluff_evidence, key, revealed_bluff)
                self._resolved_action_objects.add(observation_key)

    @staticmethod
    def _serialize(evidence: dict[str, list[int]]) -> dict[str, dict[str, int]]:
        return {key: {"successes": value[0], "failures": value[1]}
                for key, value in sorted(evidence.items())}

    @staticmethod
    def _deserialize(value: Any) -> dict[str, list[int]]:
        if not isinstance(value, dict):
            raise ValueError("malformed round adaptive evidence")
        evidence: dict[str, list[int]] = {}
        valid_bands = {_context(q, pile) for q in (1, 2, 3, 4)
                       for pile in (1, 3, 6)}
        for key, counts in value.items():
            if key not in valid_bands or not isinstance(counts, dict):
                raise ValueError("malformed round adaptive context")
            successes, failures = counts.get("successes"), counts.get("failures")
            if (type(successes) is not int or type(failures) is not int
                    or successes < 0 or failures < 0):
                raise ValueError("malformed round adaptive counts")
            evidence[key] = [successes, failures]
        return evidence

    def to_dict(self) -> dict:
        return {
            "version": _PROFILE_VERSION,
            "policy": self.policy_name,
            "ruleset_id": RULESET_ID,
            "call_evidence": self._serialize(self._call_evidence),
            "bluff_evidence": self._serialize(self._bluff_evidence),
        }

    def restore_opponent_profile(self, data: dict) -> None:
        super().restore_opponent_profile(data)
        if "call_evidence" not in data or "bluff_evidence" not in data:
            raise ValueError("incomplete round adaptive profile")
        calls = self._deserialize(data["call_evidence"])
        bluffs = self._deserialize(data["bluff_evidence"])
        self._call_evidence = calls
        self._bluff_evidence = bluffs


__all__ = [
    "RoundRandomBot", "RoundHonestBot", "RoundMathBot", "RoundAdaptiveBot",
    "_hypergeom_at_least",
]
