"""Offline two-expert online Bayes blend for binary forecasts."""

from math import exp, isfinite, log, log1p

from game_v2 import RULESET_ID


class BinaryForecastBlend:
    """Prequential Bayes blend of fixed baseline and contextual probabilities.

    This scores binary call forecasts only; it makes no claim about game wins.
    """

    VERSION = 1
    MODEL_ID = "round-binary-forecast-blend-baseline-contextual-v1"
    BASELINE = "baseline"
    CONTEXTUAL = "contextual"
    PRIOR = 0.5

    def __init__(self):
        self._losses = {self.BASELINE: 0.0, self.CONTEXTUAL: 0.0}
        self._meta_loss = 0.0
        self._updates = 0

    @staticmethod
    def _probability(value, name):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or not 0 < value < 1):
            raise ValueError(f"{name} must be finite and strictly between 0 and 1")
        return float(value)

    @staticmethod
    def _logsumexp(a, b):
        high = max(a, b)
        return high + log1p(exp(min(a, b) - high))

    def _log_weights(self):
        # Equal fixed priors cancel under normalization.
        return -self._losses[self.BASELINE], -self._losses[self.CONTEXTUAL]

    @property
    def weights(self):
        a, b = self._log_weights()
        normalizer = self._logsumexp(a, b)
        return {self.BASELINE: exp(a - normalizer),
                self.CONTEXTUAL: exp(b - normalizer)}

    @property
    def losses(self):
        return dict(self._losses)

    @property
    def meta_loss(self):
        return self._meta_loss

    @property
    def updates(self):
        return self._updates

    def predict(self, p_baseline, p_contextual):
        pb = self._probability(p_baseline, "p_baseline")
        pc = self._probability(p_contextual, "p_contextual")
        weights = self.weights
        return weights[self.BASELINE] * pb + weights[self.CONTEXTUAL] * pc

    def observe(self, called, p_baseline, p_contextual, forced=False):
        if type(called) is not bool or type(forced) is not bool:
            raise ValueError("called and forced must be bools")
        if forced and not called:
            raise ValueError("a forced response must be called")
        pb = self._probability(p_baseline, "p_baseline")
        pc = self._probability(p_contextual, "p_contextual")
        prediction = self.predict(pb, pc)  # strictly pre-update
        if forced:
            return {"probability": prediction, "updated": False}

        outcome_loss = (lambda p: -log(p)) if called else (lambda p: -log1p(-p))
        baseline_loss = outcome_loss(pb)
        contextual_loss = outcome_loss(pc)
        meta_increment = outcome_loss(prediction)
        self._losses[self.BASELINE] += baseline_loss
        self._losses[self.CONTEXTUAL] += contextual_loss
        self._meta_loss += meta_increment
        self._updates += 1
        return {"probability": prediction, "updated": True}

    def to_dict(self):
        return {"version": self.VERSION, "model_id": self.MODEL_ID,
                "ruleset_id": RULESET_ID,
                "config": {"predictors": [self.BASELINE, self.CONTEXTUAL],
                           "prior": {self.BASELINE: self.PRIOR,
                                     self.CONTEXTUAL: self.PRIOR},
                           "labels": {"called": True, "not_called": False}},
                "losses": dict(self._losses), "meta_loss": self._meta_loss,
                "updates": self._updates}

    def restore(self, profile):
        keys = {"version", "model_id", "ruleset_id", "config", "losses",
                "meta_loss", "updates"}
        expected_config = {"predictors": [self.BASELINE, self.CONTEXTUAL],
                           "prior": {self.BASELINE: self.PRIOR,
                                     self.CONTEXTUAL: self.PRIOR},
                           "labels": {"called": True, "not_called": False}}
        config = profile.get("config") if isinstance(profile, dict) else None
        if (not isinstance(profile, dict) or set(profile) != keys
                or type(profile["version"]) is not int or profile["version"] != self.VERSION
                or profile["model_id"] != self.MODEL_ID
                or profile["ruleset_id"] != RULESET_ID
                or config != expected_config
                or not isinstance(config, dict)
                or not isinstance(config.get("labels"), dict)
                or any(type(value) is not bool for value in config["labels"].values())):
            raise ValueError("invalid binary forecast blend profile")
        values = profile["losses"]
        if not isinstance(values, dict) or set(values) != {self.BASELINE, self.CONTEXTUAL}:
            raise ValueError("invalid expert losses")
        losses = {}
        for name, value in values.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or value < 0):
                raise ValueError("expert losses must be finite nonnegative numbers")
            losses[name] = float(value)
        meta_loss = profile["meta_loss"]
        updates = profile["updates"]
        if (isinstance(meta_loss, bool) or not isinstance(meta_loss, (int, float))
                or not isfinite(meta_loss) or meta_loss < 0
                or type(updates) is not int or updates < 0):
            raise ValueError("invalid meta loss or update count")
        # L_meta = -log(sum_i prior_i * exp(-L_i)); validate before mutation.
        a = log(self.PRIOR) - losses[self.BASELINE]
        b = log(self.PRIOR) - losses[self.CONTEXTUAL]
        expected_meta = -self._logsumexp(a, b)
        tolerance = 1e-9 * max(1.0, abs(expected_meta))
        if abs(float(meta_loss) - expected_meta) > tolerance:
            raise ValueError("meta loss is inconsistent with expert losses")
        self._losses, self._meta_loss, self._updates = losses, float(meta_loss), updates


__all__ = ["BinaryForecastBlend"]
