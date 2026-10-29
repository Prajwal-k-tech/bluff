"""Bayesian global-versus-context pooling for own-truth call responses."""

from math import exp, lgamma

from bots.round_normalized_defense_bot import RoundPriorOnlyEmpiricalDefenseBot
from bots.round_policy import _context


class RoundBayesianPoolResponseBot(RoundPriorOnlyEmpiricalDefenseBot):
    """Average shared and context-specific Beta call-rate models by evidence.

    Counts are only the inherited, legally observed own-packet truth response
    counts. The model comparison is performed separately for each truth label.
    """

    policy_name = "round_bayesian_pool_truth_response_research_v1"
    _alpha = 1.4
    _beta = 2.6

    def _plan(self, hand, state):
        # Candidate scoring revisits contexts. Keep marginal likelihoods local
        # to this decision so evidence can never go stale across observations.
        self._bayes_plan_cache = {}
        try:
            return super()._plan(hand, state)
        finally:
            del self._bayes_plan_cache

    @staticmethod
    def _log_beta(a, b):
        return lgamma(a) + lgamma(b) - lgamma(a + b)

    def _model_weight(self, truth_name, queried_key):
        key = (truth_name, queried_key)
        cache = getattr(self, "_bayes_plan_cache", None)
        if cache is not None and key in cache:
            return cache[key]
        cells = self.truth_calls[truth_name]
        global_success = sum(v[0] for v in cells.values())
        global_failure = sum(v[1] for v in cells.values())
        local_success, local_failure = cells.get(queried_key, (0, 0))
        prior = self._log_beta(self._alpha, self._beta)
        global_log_evidence = (self._log_beta(self._alpha + global_success,
                                              self._beta + global_failure) - prior)
        local_log_evidence = sum(
            self._log_beta(self._alpha + value[0], self._beta + value[1]) - prior
            for value in cells.values()
        )
        # Equal prior odds; logistic in log space avoids exponent overflow.
        delta = global_log_evidence - local_log_evidence
        if delta >= 0:
            global_weight = 1.0 / (1.0 + exp(-delta))
        else:
            e = exp(delta)
            global_weight = e / (1.0 + e)
        if cache is not None:
            cache[key] = global_weight
        return global_weight

    def _play_utility(self, hand, state, claimed_rank, quantity, truthful,
                      pile_after, call_probability):
        truth_name = "truth" if truthful else "bluff"
        key = _context(quantity, pile_after)
        cells = self.truth_calls[truth_name]
        global_success = sum(v[0] for v in cells.values())
        global_failure = sum(v[1] for v in cells.values())
        local_success, local_failure = cells.get(key, (0, 0))
        global_mean = (self._alpha + global_success) / (
            self._alpha + self._beta + global_success + global_failure)
        local_mean = (self._alpha + local_success) / (
            self._alpha + self._beta + local_success + local_failure)
        global_weight = self._model_weight(truth_name, key)
        learned = global_weight * global_mean + (1.0 - global_weight) * local_mean
        proof = self._certifiable_probability(hand, state, claimed_rank, quantity)
        probability = 1.0 if quantity == len(hand) else max(learned, proof)
        utility = quantity + probability * pile_after if truthful else quantity - probability * pile_after
        return utility, dict(
            call_probability=probability,
            truth_response_calls=local_success,
            truth_response_declines=local_failure,
            response_model="same_truth_beta_global_vs_context_evidence_v1",
            global_model_weight=global_weight,
        )
