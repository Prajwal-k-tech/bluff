"""Offline fixed-share filter over public-event risk hypotheses."""

from copy import deepcopy
from math import isfinite, log

from cards import Rank
from game_v2 import RULESET_ID


LABELS = ("rho_0", "rho_0.25", "rho_0.5", "rho_0.75", "rho_0.95")
FAMILIES = ("play", "pass", "challenge")
MODEL_ID = "public-event-risk-filter-v1"
VERSION = 1


class PublicRiskFilter:
    """Fixed-share Bayesian filter over supplied public-event laws."""

    model_id = MODEL_ID

    def __init__(self, share=0.02):
        if (isinstance(share, bool) or not isinstance(share, (int, float))
                or not isfinite(share) or not 0 <= share < 1):
            raise ValueError("share must be finite and in [0, 1)")
        self.share = float(share)
        self._weights = {label: 1.0 / len(LABELS) for label in LABELS}
        self.updates = 0
        self.loss = 0.0
        self._pending = None
        self._seen = set()

    @property
    def weights(self):
        return dict(self._weights)

    @property
    def prior_weights(self):
        return {label: (1.0 - self.share) * self._weights[label]
                + self.share / len(LABELS) for label in LABELS}

    @staticmethod
    def _laws(laws):
        if not isinstance(laws, dict) or set(laws) != set(LABELS):
            raise ValueError("laws must contain exactly the five risk labels")
        result = {}
        for label in LABELS:
            row = laws[label]
            if not isinstance(row, dict) or not row:
                raise ValueError("each risk law must be a nonempty dictionary")
            clean = {}
            for event, probability in row.items():
                PublicRiskFilter._event(event)
                if (isinstance(probability, bool)
                        or not isinstance(probability, (int, float))
                        or not isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError("event probabilities must be finite and nonnegative")
                clean[event] = float(probability)
            if abs(sum(clean.values()) - 1.0) > 1e-10:
                raise ValueError("each risk law must be normalized")
            result[label] = clean
        return result

    @staticmethod
    def _event(event):
        if type(event) is not tuple or len(event) != 3:
            raise ValueError("public event must be a (family, rank, quantity) tuple")
        family, rank, quantity = event
        if family == "play":
            valid = type(rank) is Rank and type(quantity) is int and 1 <= quantity <= 4
        elif family in ("pass", "challenge"):
            valid = rank is None and type(quantity) is int and quantity == 0
        else:
            valid = False
        if not valid:
            raise ValueError("invalid public event")

    def predict(self, laws):
        rows = self._laws(laws)
        prior = self.prior_weights
        events = set().union(*(row.keys() for row in rows.values()))
        return {event: sum(prior[label] * rows[label].get(event, 0.0)
                           for label in LABELS) for event in events}

    def forecast(self, index, laws):
        if type(index) is not int or index < 0 or index in self._seen:
            raise ValueError("invalid or closed event index")
        rows = self._laws(laws)
        if self._pending is not None:
            old_index, old_rows, prediction, prior = self._pending
            if old_index != index:
                raise ValueError("an earlier public event is still pending")
            if old_rows != rows:
                raise ValueError("pending public forecast changed")
            return deepcopy(prediction)
        prior = self.prior_weights
        events = set().union(*(row.keys() for row in rows.values()))
        prediction = {event: sum(prior[label] * rows[label].get(event, 0.0)
                                for label in LABELS) for event in events}
        self._pending = (index, deepcopy(rows), deepcopy(prediction), dict(prior))
        return prediction

    def observe(self, index, event, *, forced=False):
        if type(index) is not int or index < 0 or type(forced) is not bool:
            raise ValueError("invalid event metadata")
        self._event(event)
        if forced and event[0] != "challenge":
            raise ValueError("forced event must be a challenge")
        if self._pending is None or self._pending[0] != index:
            return False
        _, rows, prediction, prior = self._pending
        if forced:
            self._pending = None
            self._seen.add(index)
            return False
        probability = prediction.get(event, 0.0)
        if not isfinite(probability) or probability <= 0:
            raise ValueError("observed event has zero forecast probability")
        staged = {label: prior[label] * rows[label].get(event, 0.0) / probability
                  for label in LABELS}
        staged_loss = self.loss - log(probability)
        if (any(not isfinite(value) or value < 0 for value in staged.values())
                or not isfinite(staged_loss) or staged_loss < 0):
            raise ValueError("invalid risk-filter update")
        self._weights = staged
        self.loss = staged_loss
        self.updates += 1
        self._pending = None
        self._seen.add(index)
        return True

    def discard_forecast(self, index):
        if type(index) is not int or index < 0:
            raise ValueError("invalid event index")
        if self._pending is not None and self._pending[0] == index:
            self._pending = None
            return True
        return False

    def begin_deal(self):
        self._pending = None
        self._seen.clear()

    def reset_profile(self):
        self._weights = {label: 1.0 / len(LABELS) for label in LABELS}
        self.updates = 0
        self.loss = 0.0
        self.begin_deal()

    def to_dict(self):
        return {"version": VERSION, "model_id": self.model_id,
                "ruleset_id": RULESET_ID,
                "config": {"labels": list(LABELS), "share": float(self.share)},
                "weights": {label: float(self._weights[label]) for label in LABELS},
                "updates": int(self.updates), "loss": float(self.loss)}

    def restore(self, profile):
        if (not isinstance(profile, dict)
                or set(profile) != {"version", "model_id", "ruleset_id", "config",
                                    "weights", "updates", "loss"}
                or type(profile["version"]) is not int or profile["version"] != VERSION
                or profile["model_id"] != self.model_id
                or profile["ruleset_id"] != RULESET_ID):
            raise ValueError("unsupported public risk-filter checkpoint")
        config = profile["config"]
        if (not isinstance(config, dict) or set(config) != {"labels", "share"}
                or type(config["labels"]) is not list or config["labels"] != list(LABELS)
                or isinstance(config["share"], bool)
                or not isinstance(config["share"], (int, float))
                or not isfinite(config["share"]) or float(config["share"]) != self.share):
            raise ValueError("risk-filter configuration mismatch")
        raw = profile["weights"]
        if not isinstance(raw, dict) or set(raw) != set(LABELS):
            raise ValueError("invalid risk-filter weights")
        weights = {}
        for label in LABELS:
            value = raw[label]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or not 0 <= value <= 1):
                raise ValueError("weights must be finite and nonnegative")
            weights[label] = float(value)
        if abs(sum(weights.values()) - 1.0) > 1e-10:
            raise ValueError("risk-filter weights must sum to one")
        updates, loss = profile["updates"], profile["loss"]
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid risk-filter update count")
        if (isinstance(loss, bool) or not isinstance(loss, (int, float))
                or not isfinite(loss) or loss < 0):
            raise ValueError("invalid risk-filter loss")
        loss = float(loss)
        uniform = all(abs(weights[label] - 1.0 / len(LABELS)) <= 1e-10
                      for label in LABELS)
        if updates == 0 and (not uniform or loss != 0.0):
            raise ValueError("zero-update checkpoint must have uniform weights and zero loss")
        self._weights, self.updates, self.loss = weights, updates, loss
        self.begin_deal()


__all__ = ["LABELS", "PublicRiskFilter"]
