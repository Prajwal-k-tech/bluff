"""Frozen empirical response-family actor for research search continuations.

The actor samples a smoothed public play/pass/challenge family forecast, then
uses a declared uniform conditional law over legal play quantities and physical
packets. This is a heuristic actor law, not a Bayesian opponent identity model.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite

from bots.round_policy import (
    RoundMathBot, _RoundPolicy, _can_prove_false, _latest_action,
    _packet_size,
)
from bots.round_response_model import RoundResponseModel
from cards import Rank
from game_v2 import RULESET_ID


_PROFILE_VERSION = 1
_FAMILIES = ("play", "pass", "challenge")


class RoundResponseFamilyActor(_RoundPolicy):
    """A frozen-count actor using only its own hand and legal public view."""

    policy_name = "round_response_family_actor_research"
    actor_law = (
        "smoothed empirical response-family forecast; mask and renormalize "
        "legal families; uniform legal quantity then uniform physical subset; "
        "Math opening baseline; truthful terminal finish and provable-false "
        "challenge safeguards override sampled family; no branch learning"
    )

    def __init__(self, response_model_profile=None, seed=None):
        super().__init__(seed)
        self._response_model = RoundResponseModel()
        if response_model_profile is not None:
            self._response_model.restore_opponent_profile(
                deepcopy(response_model_profile))
        self._math = RoundMathBot()
        self._math.rng = self.rng

    def reset(self):
        super().reset()
        self._math.reset()
        self._math.rng = self.rng
        self._math.player_id = self.player_id

    def observe_action(self, action, opponent_hand_size):
        """Keep the Math opening baseline's conservative public bookkeeping."""
        self._math.player_id = self.player_id
        self._math.rng = self.rng
        self._math.observe_action(action, opponent_hand_size)

    def _family_probabilities(self, hand, state, legal_families):
        latest = _latest_action(state)
        if latest is None:
            raise ValueError("response family forecast requires a latest claim")
        quantity = _packet_size(latest)
        pile_size = state.get("pile_size")
        response_hand_size = len(hand)
        probabilities = self._response_model.predict(
            quantity, pile_size, response_hand_size)
        masked = {}
        for legal_family in legal_families:
            model_family = {"call_bluff": "challenge"}.get(
                legal_family, legal_family)
            masked[model_family] = probabilities[model_family]
        total = sum(masked.values())
        if not isfinite(total) or total <= 0:
            raise ValueError("legal response families have no forecast support")
        return {family: value / total for family, value in masked.items()}

    def _sample_family(self, probabilities):
        threshold = self.rng.random()
        cumulative = 0.0
        for family in _FAMILIES:
            cumulative += probabilities.get(family, 0.0)
            if threshold < cumulative:
                return family
        # Floating point normalization can leave a final ulp; deterministic
        # fallback uses the last supported family in canonical order.
        return next(family for family in reversed(_FAMILIES)
                    if probabilities.get(family, 0.0) > 0)

    @staticmethod
    def _truthful_finish(hand, state, legal_families):
        active = state.get("active_rank")
        if ("play" not in legal_families or active is None or not hand
                or len(hand) > 4 or not isinstance(active, Rank)):
            return False
        return all(card.rank == active for card in hand)

    def choose_action(self, hand, state):
        hand = list(hand)
        self._set_viewer(state)
        self._math.player_id = self.player_id
        self._math.rng = self.rng

        # Opening choices use the declared fixed Math baseline. There is no
        # active response-family context in a newly started round.
        if state.get("active_rank") is None:
            choice = self._math.choose_action(hand, state)
            self.last_decision_telemetry = {
                "policy": self.policy_name,
                "actor_law": self.actor_law,
                "action": choice["action"],
                "opening_model": "RoundMathBot",
            }
            return choice

        legal_families = self._legal_families(hand, state)
        if self._truthful_finish(hand, state, legal_families):
            active = state["active_rank"]
            choice = self._play(list(hand), active)
            self.last_decision_telemetry = {
                "policy": self.policy_name,
                "actor_law": self.actor_law,
                "action": "play",
                "safeguard": "truthful_terminal_finish",
                "quantity": len(hand),
                "rank": int(active),
            }
            return choice

        latest = _latest_action(state)
        if ("call_bluff" in legal_families
                and latest is not None and _can_prove_false(hand, state, latest)):
            self.last_decision_telemetry = {
                "policy": self.policy_name,
                "actor_law": self.actor_law,
                "action": "call_bluff",
                "safeguard": "provably_false_latest_packet",
            }
            return self._simple("call_bluff")

        probabilities = self._family_probabilities(hand, state, legal_families)
        family = self._sample_family(probabilities)
        if family == "challenge":
            choice = self._simple("call_bluff")
        elif family == "pass":
            choice = self._simple("pass")
        else:
            quantities = tuple(range(1, min(4, len(hand)) + 1))
            if not quantities:
                raise ValueError("play family has no legal nonempty packet")
            quantity = self.rng.choice(quantities)
            packet = self.rng.sample(hand, quantity)
            active = state.get("active_rank")
            if not isinstance(active, Rank):
                raise ValueError("continuation requires a canonical active rank")
            choice = self._play(packet, active)

        self.last_decision_telemetry = {
            "policy": self.policy_name,
            "actor_law": self.actor_law,
            "action": choice["action"],
            "response_family_probabilities": probabilities,
            "quantity": len(choice.get("cards", ())),
            "rank": (int(choice["rank"]) if choice.get("rank") is not None else None),
            "frozen_response_model": True,
        }
        return choice

    def to_dict(self):
        return {"version": _PROFILE_VERSION,
                "policy": self.policy_name,
                "ruleset_id": RULESET_ID,
                "response_model": deepcopy(self._response_model.to_dict())}

    def restore_opponent_profile(self, data):
        if (not isinstance(data, dict)
                or set(data) != {"version", "policy", "ruleset_id", "response_model"}
                or type(data.get("version")) is not int
                or data.get("version") != _PROFILE_VERSION
                or data.get("policy") != self.policy_name
                or data.get("ruleset_id") != RULESET_ID):
            raise ValueError("unsupported response-family actor profile")
        candidate = RoundResponseModel()
        candidate.restore_opponent_profile(deepcopy(data["response_model"]))
        self._response_model = candidate


__all__ = ["RoundResponseFamilyActor"]
