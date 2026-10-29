"""Stateless, information-safe toy style actors for corrected-round search."""

from collections import Counter
from math import comb, isfinite
from random import Random

from bots.round_continuation_likelihood import continuation_distribution
from bots.round_opening_likelihood import STYLES, opening_claim_distribution
from bots.round_policy import _RoundPolicy, _can_prove_false, _packet_size
from cards import Rank
from game_v2 import RULESET_ID


class RoundGenerativeStyleBot(_RoundPolicy):
    """Sample a style's event model and then a uniformly compatible packet.

    The actor reads only ``hand`` and its redacted ``round_player_view``. It
    has no learned profile; ``observe_action`` is intentionally a no-op.
    """

    policy_name = "round_generative_style"

    def __init__(self, style="mixed_uniform", seed=None, *,
                 bluff_probability=0.3, call_probability=0.35,
                 mixed_respects_certificates=False,
                 mixed_takes_truthful_finish=False):
        super().__init__(seed)
        if style not in STYLES:
            raise ValueError("style must be one of the canonical styles")
        self.style = style
        self.bluff_probability = self._probability(bluff_probability, "bluff_probability")
        self.call_probability = self._probability(call_probability, "call_probability")
        if type(mixed_respects_certificates) is not bool:
            raise TypeError("mixed_respects_certificates must be a bool")
        self.mixed_respects_certificates = mixed_respects_certificates
        if type(mixed_takes_truthful_finish) is not bool:
            raise TypeError("mixed_takes_truthful_finish must be a bool")
        self.mixed_takes_truthful_finish = mixed_takes_truthful_finish

    @staticmethod
    def _probability(value, name):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return float(value)

    def _sample_event(self, distribution):
        events = [(event, mass) for event, mass in distribution.items() if mass > 0]
        total = sum(mass for _, mass in events)
        if not total:
            raise ValueError("style has no support in this legal observation")
        threshold = self.rng.random() * total
        for event, mass in events:
            threshold -= mass
            if threshold < 0:
                return event
        return events[-1][0]

    def _packet(self, hand, rank, quantity, truthful):
        matching = [card for card in hand if card.rank == rank]
        if truthful:
            if len(matching) < quantity:
                raise ValueError("truthful event is incompatible with actor hand")
            return self.rng.sample(matching, quantity)
        others = [card for card in hand if card.rank != rank]
        n, other_n = len(matching), len(others)
        # P(m matching | q cards, at least one mismatch), proportional to
        # C(n,m) C(N-n,q-m), exactly the uniform-subset conditional law.
        support = []
        for m in range(max(0, quantity - other_n), min(n, quantity - 1) + 1):
            weight = comb(n, m) * comb(other_n, quantity - m)
            if weight:
                support.append((m, weight))
        total = sum(weight for _, weight in support)
        if not total:
            raise ValueError("false event is incompatible with actor hand")
        pick = self.rng.randrange(total)
        matching_count = support[-1][0]
        for m, weight in support:
            pick -= weight
            if pick < 0:
                matching_count = m
                break
        return (self.rng.sample(matching, matching_count)
                + self.rng.sample(others, quantity - matching_count))

    def choose_action(self, hand, state):
        hand = list(hand)
        self._set_viewer(state)
        families = self._legal_families(hand, state)
        active = state.get("active_rank")
        sampled_truth = None
        if active is None:
            distribution = opening_claim_distribution(
                Counter(card.rank for card in hand), self.style,
                self.bluff_probability,
                mixed_takes_truthful_finish=self.mixed_takes_truthful_finish)
            # Opening events already include rank, quantity, and truth.
            supported = {event: mass for event, mass in distribution.items()
                         if "play" in families}
            if not supported:
                raise ValueError("no legal opening play is supported")
            rank, quantity, truthful = self._sample_event(supported)
            sampled_truth = truthful
            result = self._play(self._packet(hand, rank, quantity, truthful), rank)
        else:
            if not isinstance(active, Rank):
                raise ValueError("active_rank must be a canonical Rank")
            last = state.get("last_action")
            if last is None:
                raise ValueError("continuation requires a latest claim")
            own_pile_count = sum(card.rank == active
                                 for card in state.get("own_pile_cards", ()))
            other_certified_false = _can_prove_false(hand, state, last)
            distribution = continuation_distribution(
                Counter(card.rank for card in hand), self.style, active,
                _packet_size(last), own_pile_rank_count=own_pile_count,
                other_certified_false=other_certified_false,
                mixed_respects_certificates=self.mixed_respects_certificates,
                mixed_takes_truthful_finish=self.mixed_takes_truthful_finish,
                bluff_probability=self.bluff_probability,
                call_probability=self.call_probability)
            supported = {}
            for event, mass in distribution.items():
                kind, quantity, truthful = event
                if ((kind == "play" and "play" in families)
                        or (kind == "pass" and "pass" in families)
                        or (kind == "call_bluff" and "call_bluff" in families)):
                    supported[event] = mass
            kind, quantity, truthful = self._sample_event(supported)
            if kind == "play":
                sampled_truth = truthful
                result = self._play(self._packet(hand, active, quantity, truthful), active)
            else:
                result = self._simple(kind)
        self.last_decision_telemetry = {
            "policy": self.policy_name, "style": self.style,
            "action": result["action"], "legal_families": tuple(families),
            "sampled_truth": sampled_truth,
        }
        return result

    def to_dict(self):
        profile = {
            "version": 1, "policy": self.policy_name,
            "ruleset_id": RULESET_ID, "style": self.style,
            "bluff_probability": self.bluff_probability,
            "call_probability": self.call_probability,
        }
        if self.mixed_respects_certificates:
            profile["model_id"] = "round_generative_style_mixed_certificate_v1"
            profile["mixed_respects_certificates"] = True
        if self.mixed_takes_truthful_finish:
            profile["model_id"] = "round_generative_style_mixed_finish_v1"
            profile["mixed_takes_truthful_finish"] = True
        return profile

    def restore_opponent_profile(self, data):
        expected = self.to_dict()
        if not isinstance(data, dict) or set(data) != set(expected):
            raise ValueError("invalid generative-style profile schema")
        if self.mixed_respects_certificates and (
                data.get("model_id") != expected["model_id"]
                or type(data.get("mixed_respects_certificates")) is not bool
                or data["mixed_respects_certificates"] is not True):
            raise ValueError("generative-style model configuration mismatch")
        if self.mixed_takes_truthful_finish and (
                data.get("model_id") != expected["model_id"]
                or type(data.get("mixed_takes_truthful_finish")) is not bool
                or data["mixed_takes_truthful_finish"] is not True):
            raise ValueError("generative-style finish configuration mismatch")
        if (type(data.get("version")) is not int or data["version"] != 1
                or data.get("policy") != self.policy_name
                or data.get("ruleset_id") != RULESET_ID
                or data.get("style") != self.style
                or type(data.get("bluff_probability")) not in (int, float)
                or isinstance(data.get("bluff_probability"), bool)
                or not isfinite(data["bluff_probability"])
                or data["bluff_probability"] != self.bluff_probability
                or type(data.get("call_probability")) not in (int, float)
                or isinstance(data.get("call_probability"), bool)
                or not isfinite(data["call_probability"])
                or data["call_probability"] != self.call_probability):
            raise ValueError("generative-style profile configuration mismatch")


__all__ = ["RoundGenerativeStyleBot"]
