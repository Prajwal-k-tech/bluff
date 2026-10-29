"""Research posterior for call propensity conditioned on proof status."""

from math import isfinite

from game_v2 import RULESET_ID


_GRID_SIZE = 21
_GRID = tuple((i + 0.5) / _GRID_SIZE for i in range(_GRID_SIZE))
_PRIOR = (1 / (_GRID_SIZE * _GRID_SIZE),) * (_GRID_SIZE * _GRID_SIZE)


def _unit(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


class ProofConditionedCallPosterior:
    """Joint finite Bayes model for call given proof and other evidence.

    ``certificate_mass`` estimates the share of call decisions with a legal
    estimate of proof. Declining remains possible for every certificate mass.
    Restart share mixes the updated posterior with the joint uniform prior; it
    is robustness smoothing, not a changepoint detector.
    """

    VERSION = 1
    MODEL_ID = "proof-conditioned-call-joint-grid21-v1"

    def __init__(self, restart_share=0.0):
        self._restart_share = _unit(restart_share, "restart_share")
        self._weights = _PRIOR
        self._updates = 0

    @property
    def weights(self):
        return tuple(self._weights)

    @property
    def means(self):
        proof = other = 0.0
        for i, theta_proof in enumerate(_GRID):
            for j, theta_other in enumerate(_GRID):
                weight = self._weights[i * _GRID_SIZE + j]
                proof += theta_proof * weight
                other += theta_other * weight
        return proof, other

    @property
    def updates(self):
        return self._updates

    def predict(self, certificate_mass):
        c = _unit(certificate_mass, "certificate_mass")
        proof, other = self.means
        return c * proof + (1 - c) * other

    def observe(self, called, certificate_mass, forced=False):
        if type(called) is not bool or type(forced) is not bool:
            raise ValueError("called and forced must be bools")
        c = _unit(certificate_mass, "certificate_mass")
        if forced and not called:
            raise ValueError("a forced final response must be called")
        probability = self.predict(c)
        likelihood = probability if called else 1 - probability
        if forced:
            return {"probability": probability, "event_likelihood": 1.0,
                    "updated": False}

        raw = tuple(weight * ((c * _GRID[i // _GRID_SIZE]
                              + (1 - c) * _GRID[i % _GRID_SIZE])
                             if called else
                             (1 - (c * _GRID[i // _GRID_SIZE]
                                   + (1 - c) * _GRID[i % _GRID_SIZE])))
                    for i, weight in enumerate(self._weights))
        total = sum(raw)
        if total <= 0 or not isfinite(total):
            raise ValueError("observed response has zero likelihood")
        posterior = tuple(value / total for value in raw)
        share = self._restart_share
        self._weights = tuple((1 - share) * value + share * prior
                              for value, prior in zip(posterior, _PRIOR))
        self._updates += 1
        return {"probability": probability, "event_likelihood": likelihood,
                "updated": True}

    def to_dict(self):
        return {"version": self.VERSION, "model_id": self.MODEL_ID,
                "ruleset_id": RULESET_ID,
                "config": {"grid_size": _GRID_SIZE,
                           "restart_share": self._restart_share},
                "weights": list(self._weights), "updates": self._updates}

    def restore(self, profile):
        keys = {"version", "model_id", "ruleset_id", "config", "weights", "updates"}
        if (not isinstance(profile, dict) or set(profile) != keys
                or type(profile["version"]) is not int or profile["version"] != self.VERSION
                or profile["model_id"] != self.MODEL_ID
                or profile["ruleset_id"] != RULESET_ID):
            raise ValueError("invalid proof-conditioned call profile schema")
        config = profile["config"]
        if (not isinstance(config, dict) or set(config) != {"grid_size", "restart_share"}
                or type(config["grid_size"]) is not int or config["grid_size"] != _GRID_SIZE):
            raise ValueError("invalid proof-conditioned call config")
        share = _unit(config["restart_share"], "restart_share")
        values = profile["weights"]
        if (not isinstance(values, list) or len(values) != _GRID_SIZE ** 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not isfinite(v) or v < 0 for v in values)):
            raise ValueError("invalid proof-conditioned call weights")
        weights = tuple(float(v) for v in values)
        if abs(sum(weights) - 1) > 1e-10:
            raise ValueError("proof-conditioned call weights must sum to one")
        updates = profile["updates"]
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid proof-conditioned call update count")
        self._restart_share, self._weights, self._updates = share, weights, updates


__all__ = ["ProofConditionedCallPosterior"]
