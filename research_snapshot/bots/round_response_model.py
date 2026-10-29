"""Research-only predictor of public response-family choices.

It uses a strength-4 uniform family prior, strength-4 global/local empirical
Bayes shrinkage, and .98 decay across 27 contexts. The global mean includes the
local counts. This heuristic is not an exact posterior or optimal policy; it
does not model card beliefs or generate full actions.
"""

from math import isfinite

from bots.round_policy import _context, _field
from game_v2 import RULESET_ID


_VERSION = 1
_DECAY = 0.98
_FAMILIES = ("play", "pass", "challenge")


def _hand_band(size):
    return "1-4" if size <= 4 else "5-12" if size <= 12 else "13+"


def _context_key(quantity, pile_size, response_hand_size):
    return f"{_context(quantity, pile_size)}|{_hand_band(response_hand_size)}"


_VALID_CONTEXTS = tuple(sorted({
    _context_key(quantity, pile_size, hand_size)
    for quantity in range(1, 5)
    for pile_size in (1, 3, 6)
    for hand_size in (1, 5, 13)
}))


def _valid_int(value):
    return type(value) is int


class RoundResponseModel:
    """Predict play/pass/challenge rates from accepted public responses only."""

    policy_name = "round_response_model_research"

    def __init__(self):
        self._counts = {key: [0.0, 0.0, 0.0] for key in _VALID_CONTEXTS}
        self._seen_action_indices = set()

    @staticmethod
    def _validate_inputs(quantity, pile_size, response_hand_size):
        if (not _valid_int(quantity) or not 1 <= quantity <= 4
                or not _valid_int(pile_size) or not quantity <= pile_size <= 52
                or not _valid_int(response_hand_size)
                or not 1 <= response_hand_size <= 52):
            raise ValueError("invalid response-model prediction context")

    def predict(self, quantity, pile_size, response_hand_size):
        """Return normalized heuristic probabilities for the three families."""
        self._validate_inputs(quantity, pile_size, response_hand_size)
        local_key = _context_key(quantity, pile_size, response_hand_size)
        global_counts = [sum(counts[index] for counts in self._counts.values())
                         for index in range(3)]
        global_total = sum(global_counts)
        global_mean = [(4 / 3 + count) / (4 + global_total)
                       for count in global_counts]
        local = self._counts[local_key]
        local_total = sum(local)
        probabilities = [(4 * global_mean[index] + local[index])
                         / (4 + local_total) for index in range(3)]
        total = sum(probabilities)
        return {family: probabilities[index] / total
                for index, family in enumerate(_FAMILIES)}

    @staticmethod
    def _packet_quantity(claim):
        quantity = _field(claim, "quantity")
        if _valid_int(quantity):
            return quantity
        cards = _field(claim, "cards_played")
        try:
            return len(cards)
        except TypeError:
            return None

    def observe_action(self, public_claim, viewer):
        """Accept one resolved response record, without consulting hidden labels."""
        if not _valid_int(viewer) or viewer not in (0, 1):
            return False
        player = _field(public_claim, "player")
        response_player = _field(public_claim, "response_player")
        forced = _field(public_claim, "response_was_forced")
        kind = _field(public_claim, "response_kind")
        was_call = _field(public_claim, "response_was_call")
        action_index = _field(public_claim, "action_index")
        quantity = self._packet_quantity(public_claim)
        pile_before = _field(public_claim, "pile_size_before")
        response_hand_size = _field(public_claim, "response_hand_size_before")
        if (not _valid_int(player) or player != viewer
                or not _valid_int(response_player) or response_player != 1 - viewer
                or forced is not False
                or kind not in _FAMILIES or type(was_call) is not bool
                or was_call is not (kind == "challenge")
                or not _valid_int(action_index) or action_index < 0
                or not _valid_int(quantity) or not 1 <= quantity <= 4
                or not _valid_int(pile_before) or pile_before < 0
                or pile_before + quantity > 52
                or not _valid_int(response_hand_size)
                or not 1 <= response_hand_size <= 52):
            return False
        if action_index in self._seen_action_indices:
            return False

        key = _context_key(quantity, pile_before + quantity, response_hand_size)
        for counts in self._counts.values():
            for index in range(3):
                counts[index] *= _DECAY
        self._counts[key][_FAMILIES.index(kind)] += 1.0
        self._seen_action_indices.add(action_index)
        return True

    def reset_deal(self):
        """Allow new action indices while preserving learned response counts."""
        self._seen_action_indices.clear()

    def reset_profile(self):
        self._counts = {key: [0.0, 0.0, 0.0] for key in _VALID_CONTEXTS}
        self._seen_action_indices.clear()

    def to_dict(self):
        return {
            "version": _VERSION,
            "policy": self.policy_name,
            "ruleset_id": RULESET_ID,
            "counts": {key: list(counts) for key, counts in self._counts.items()},
        }

    def restore_opponent_profile(self, data):
        if (not isinstance(data, dict)
                or set(data) != {"version", "policy", "ruleset_id", "counts"}
                or type(data.get("version")) is not int
                or data.get("version") != _VERSION
                or data.get("policy") != self.policy_name
                or data.get("ruleset_id") != RULESET_ID
                or not isinstance(data.get("counts"), dict)
                or set(data["counts"]) != set(_VALID_CONTEXTS)):
            raise ValueError("unsupported response-model profile")
        candidate = {}
        mass = 0.0
        for key, values in data["counts"].items():
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(value) not in (int, float) or not isfinite(value)
                           or value < 0 for value in values)):
                raise ValueError("malformed response-model counts")
            for value in values:
                if value > 50 + 1e-8 - mass:
                    raise ValueError("response-model count mass exceeds decay bound")
                mass += value
            candidate[key] = [float(value) for value in values]
        self._counts = candidate
        self._seen_action_indices.clear()
