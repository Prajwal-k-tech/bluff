"""Unpromoted response mixture with explicit categorical model-mismatch support.

Finite approximate world samples can produce false zero likelihoods. The fixed
5% uniform contamination is a declared research assumption, not a calibrated
sampling interval, and applies equally to forecasts and Bayesian updates.
"""
from bots.round_counterlearning_response_learner import CounterlearningResponseLearner


class RobustResponseLearner(CounterlearningResponseLearner):
    model_id = "legal-policy-response-mixture-public-adaptive-robust05-v3"
    contamination = .05

    @staticmethod
    def _likelihoods(experts):
        rows = CounterlearningResponseLearner._likelihoods(experts)
        for row in rows.values():
            row["family"] = {family: .95*probability + .05/3
                             for family, probability in row["family"].items()}
        return rows
