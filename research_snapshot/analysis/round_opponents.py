"""Evaluation-only scripted controls, for mechanism diagnostics only.

Matching context bands do not establish held-out generalization or human likeness.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from cards import Card, Rank
from game_v2 import RULESET_ID
from bots.round_policy import (RoundHonestBot, _latest_action, _rank,
                               _can_prove_false, _packet_size)


_DEFAULT = (0.35,) * 9
_BLUFF_DEFAULT = (0.30,) * 9


def _probabilities(value, name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != 9:
        raise ValueError(f"{name} must be an immutable tuple of exactly nine probabilities")
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not 0 <= p <= 1 or not math.isfinite(p) for p in value):
        raise ValueError(f"{name} entries must be finite probabilities in [0, 1]")
    return tuple(float(p) for p in value)


@dataclass(frozen=True)
class BehaviorPattern:
    """Call/bluff probabilities by quantity band × resulting pile-size band."""

    call_probabilities: tuple[float, ...] = _DEFAULT
    bluff_probabilities: tuple[float, ...] = _BLUFF_DEFAULT

    def __post_init__(self):
        object.__setattr__(self, "call_probabilities",
                           _probabilities(self.call_probabilities, "call_probabilities"))
        object.__setattr__(self, "bluff_probabilities",
                           _probabilities(self.bluff_probabilities, "bluff_probabilities"))


class ScriptedRoundOpponent(RoundHonestBot):
    """A seeded, transparent behavioral control; never part of the server roster."""

    policy_name = "scripted_round_opponent"

    def __init__(self, pattern: Optional[BehaviorPattern] = None,
                 seed: Optional[int] = None, certify_calls: bool = True):
        if pattern is not None and not isinstance(pattern, BehaviorPattern):
            raise TypeError("pattern must be a BehaviorPattern")
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be an integer or None")
        if type(certify_calls) is not bool:
            raise TypeError("certify_calls must be a bool")
        super().__init__(seed)
        self.pattern = pattern if pattern is not None else BehaviorPattern()
        self.initial_seed = seed
        self.certify_calls = certify_calls

    def configuration(self) -> dict:
        return {
            "policy": self.policy_name,
            "ruleset_id": RULESET_ID,
            "seed": self.initial_seed,
            "certify_calls": self.certify_calls,
            "pattern": {
                "call_probabilities": list(self.pattern.call_probabilities),
                "bluff_probabilities": list(self.pattern.bluff_probabilities),
            },
        }

    def set_pattern(self, pattern: BehaviorPattern) -> None:
        if not isinstance(pattern, BehaviorPattern):
            raise TypeError("pattern must be a BehaviorPattern")
        self.pattern = pattern

    @staticmethod
    def _terminal_truthful_finish(hand: list[Card], state: dict) -> Optional[dict]:
        if not hand or len(hand) > 4 or len({card.rank for card in hand}) != 1:
            return None
        rank = hand[0].rank
        active = _rank(state.get("active_rank"))
        if active is None or active == rank:
            return RoundHonestBot._play(list(hand), rank)
        return None

    @staticmethod
    def _pile_band(size: int) -> int:
        return 0 if size <= 2 else 1 if size <= 5 else 2

    @staticmethod
    def _quantity_band(quantity: int) -> int:
        return 0 if quantity == 1 else 1 if quantity == 2 else 2

    def _probability(self, probabilities: tuple[float, ...], quantity: int,
                     pile_size: int) -> float:
        return probabilities[self._quantity_band(quantity) * 3 + self._pile_band(pile_size)]

    def choose_action(self, hand: list[Card], state: dict) -> dict:
        self._set_viewer(state)
        families = self._legal_families(hand, state)
        terminal = self._terminal_truthful_finish(hand, state) if "play" in families else None
        if terminal is not None:
            result = terminal
            reason = "truthful_terminal_finish"
            call_probability = bluff_probability = None
            quantity = len(hand)
            requested_bluff = None
            actual_bluff = False
            truthful_available, bluff_available = True, False
        else:
            last = _latest_action(state)
            call_probability = bluff_probability = None
            if "call_bluff" in families:
                if self.certify_calls and last is not None and _can_prove_false(hand, state, last):
                    result, reason = self._simple("call_bluff"), "certified_contradiction"
                    quantity = None
                    requested_bluff = actual_bluff = None
                    truthful_available = bluff_available = None
                    return self._record(result, reason, None, None, quantity,
                                        requested_bluff, truthful_available,
                                        bluff_available, actual_bluff)
                q = max(1, min(4, _packet_size(last)))
                pile = max(0, int(state.get("pile_size", q)))
                call_probability = self._probability(self.pattern.call_probabilities, q, pile)
                if self.rng.random() < call_probability:
                    result, reason = self._simple("call_bluff"), "configured_call"
                    return self._record(result, reason, call_probability, None, None,
                                        None, None, None, None)

            if "play" not in families:
                if "pass" in families:
                    result, reason = self._simple("pass"), "no_play_available"
                elif "call_bluff" in families:
                    result, reason = self._simple("call_bluff"), "sole_legal_action"
                else:
                    raise ValueError("no legal action available")
                return self._record(result, reason, call_probability, None, None,
                                    None, None, None, None)

            quantity = self.rng.randint(1, min(4, len(hand) - 1)) if len(hand) > 1 else 1
            active = _rank(state.get("active_rank"))
            honest = self._honest_play(hand, state)
            rank = active if active is not None else honest["rank"]
            matching = [card for card in hand if card.rank == rank]
            offrank = [card for card in hand if card.rank != rank]
            truthful_available = len(matching) >= quantity
            bluff_available = bool(offrank)
            pile_after = max(0, int(state.get("pile_size", 0))) + quantity
            bluff_probability = self._probability(
                self.pattern.bluff_probabilities, quantity, pile_after)
            requested_bluff = self.rng.random() < bluff_probability

            if quantity == len(hand) and not truthful_available and "pass" in families:
                result, reason = self._simple("pass"), "avoid_false_terminal"
                actual_bluff = None
                return self._record(result, reason, call_probability,
                                    bluff_probability, quantity, requested_bluff,
                                    truthful_available, bluff_available, actual_bluff)
            if requested_bluff and bluff_available:
                # Deliberately biased subset: one uniform off-rank card plus
                # uniform remaining cards, ensuring at least one mismatch.
                chosen = [self.rng.choice(offrank)]
                remaining = [card for card in hand if card not in chosen]
                chosen.extend(self.rng.sample(remaining, quantity - 1))
                result, reason = self._play(chosen, rank), "configured_bluff"
            elif truthful_available:
                result = self._play(self.rng.sample(matching, quantity), rank)
                reason = "requested_bluff_unavailable" if requested_bluff else "configured_truth"
            elif "pass" in families:
                result, reason = self._simple("pass"), "truth_unavailable"
                actual_bluff = None
                return self._record(result, reason, call_probability,
                                    bluff_probability, quantity, requested_bluff,
                                    truthful_available, bluff_available, actual_bluff)
            elif bluff_available:
                chosen = [self.rng.choice(offrank)]
                remaining = [card for card in hand if card not in chosen]
                chosen.extend(self.rng.sample(remaining, quantity - 1))
                result, reason = self._play(chosen, rank), "forced_bluff_fallback"
            else:
                raise ValueError("no legal packet available")
            actual_bluff = any(card.rank != rank for card in result["cards"])
        return self._record(result, reason, call_probability, bluff_probability,
                            quantity, requested_bluff, truthful_available,
                            bluff_available, actual_bluff)

    def _record(self, result, reason, call_probability, bluff_probability,
                quantity, requested_bluff, truthful_available, bluff_available,
                actual_bluff):
        self.last_decision_telemetry = {
            "policy": self.policy_name, "action": result["action"], "reason": reason,
            "requested_call_probability": call_probability,
            "requested_bluff_probability": bluff_probability,
            "proposed_quantity": quantity, "requested_bluff": requested_bluff,
            "truthful_available": truthful_available, "bluff_available": bluff_available,
            "actual_bluff": actual_bluff,
        }
        return result
