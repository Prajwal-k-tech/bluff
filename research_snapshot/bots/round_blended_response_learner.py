"""Prequential proof conditioned blend over the robust response mixture."""
from copy import deepcopy

from bots.round_binary_forecast_blend import BinaryForecastBlend
from bots.round_proof_conditioned_call import ProofConditionedCallPosterior
from bots.round_robust_response_learner import RobustResponseLearner
from game_v2 import RULESET_ID


class BlendedResponseLearner(RobustResponseLearner):
    model_id = "legal-policy-response-proof-blend-v1"
    meta_class = BinaryForecastBlend
    proof_class = ProofConditionedCallPosterior

    def __init__(self):
        super().__init__()
        self.proof = self.proof_class()
        self.meta = self.meta_class()
        self._blend_pending = {}

    def begin_deal(self):
        super().begin_deal()
        if hasattr(self, "_blend_pending"):
            self._blend_pending.clear()

    def _parts(self, experts):
        global_forecast = super().predict(experts)
        certificate_mass = experts["honest"]["challenge"]
        proof_call = self.proof.predict(certificate_mass)
        challenge = self.meta.predict(global_forecast["challenge"], proof_call)
        remainder = 1.0 - global_forecast["challenge"]
        if remainder <= 0:
            # Robust contamination guarantees support, but keep this explicit
            # for future parent likelihood implementations.
            raise ValueError("global response forecast has no non-challenge mass")
        scale = (1.0 - challenge) / remainder
        prediction = {family: global_forecast[family] * scale
                      for family in ("play", "pass")}
        prediction["challenge"] = challenge
        return prediction, certificate_mass, global_forecast["challenge"], proof_call

    def predict(self, experts):
        return self._parts(experts)[0]

    def forecast(self, index, experts):
        parts = self._parts(experts)
        prediction = super().forecast(index, experts)
        cached = (parts[1], parts[2], parts[3])
        prior = self._blend_pending.get(index)
        if prior is not None and prior != cached:
            raise ValueError("pending forecast changed")
        self._blend_pending.setdefault(index, cached)
        return prediction

    def observe(self, index, family, *, forced=False):
        # Validate every layer before any of the three learners mutates.
        if type(index) is not int or family not in ("play", "pass", "challenge") or type(forced) is not bool:
            raise ValueError("invalid response metadata")
        if forced and family != "challenge":
            raise ValueError("forced response must be a challenge")
        if index not in self._pending:
            return False
        if index not in self._blend_pending:
            raise ValueError("missing cached proof forecast")
        certificate_mass, baseline_call, proof_call = self._blend_pending[index]
        called = family == "challenge"
        # Child APIs validate inputs before updating. Forced calls are validated
        # too, but all three pending records are still consumed without learning.
        self.meta._probability(baseline_call, "p_baseline")
        self.meta._probability(proof_call, "p_contextual")
        self.proof.predict(certificate_mass)

        if forced:
            updated = super().observe(index, family, forced=True)
            self._blend_pending.pop(index, None)
            return updated

        self.meta.observe(called, baseline_call, proof_call)
        self.proof.observe(called, certificate_mass)
        updated = super().observe(index, family)
        self._blend_pending.pop(index, None)
        return updated

    def discard_forecast(self, index):
        discarded = super().discard_forecast(index)
        self._blend_pending.pop(index, None)
        return discarded

    def to_dict(self):
        return {"model_id": self.model_id, "ruleset_id": RULESET_ID,
                "response": super().to_dict(), "proof": self.proof.to_dict(),
                "meta": self.meta.to_dict()}

    def restore(self, profile):
        if (not isinstance(profile, dict)
                or set(profile) != {"model_id", "ruleset_id", "response", "proof", "meta"}
                or profile["model_id"] != self.model_id
                or profile["ruleset_id"] != RULESET_ID):
            raise ValueError("unsupported blended response profile")
        staged_response = RobustResponseLearner()
        staged_response.model_id = self.model_id
        staged_response.restore(deepcopy(profile["response"]))
        staged_proof = self.proof_class()
        staged_proof.restore(deepcopy(profile["proof"]))
        if staged_proof.to_dict()["config"] != self.proof.to_dict()["config"]:
            raise ValueError("blended proof model configuration changed")
        staged_meta = self.meta_class()
        staged_meta.restore(deepcopy(profile["meta"]))
        if not (staged_response.updates == staged_proof.updates == staged_meta.updates):
            raise ValueError("blended learner update counts disagree")
        self.weights, self.updates = staged_response.weights, staged_response.updates
        self.proof, self.meta = staged_proof, staged_meta
        self.begin_deal()


__all__ = ["BlendedResponseLearner"]
