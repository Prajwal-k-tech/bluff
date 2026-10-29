"""Experiment-only corrected policies; never imported by the web roster."""

from math import exp, isfinite, log

from bots.round_policy import (RoundAdaptiveBot, _can_prove_false, _context,
                               _field, _packet_size)


def _normalized(weights):
    total = sum(weights)
    return tuple(weight / total for weight in weights)


class RoundLatentPropensityBot(RoundAdaptiveBot):
    """Research-only likelihood-consistent feasibility-conditioned learner.

    Given ability H, the simplified claim model has false mass theta and
    true mass (1-theta)*H. Infer theta from Bernoulli likelihoods of PUBLIC
    voluntary reveals, rather than interpreting observed label rates as theta.
    Opening/continuation have separate posteriors. This is still NOT a full
    action-selection model: H is approximate, quantity/rank/pass selection and
    informative reveal selection within omitted contexts remain misspecified.
    A fixed log-odds quadrature approximates Beta(1.2,2.8); .02 posterior
    restart after each labelled claim permits recovery from behavior changes.
    Offensive learning and planner remain unchanged. Never in the web roster.
    """

    policy_name = "round_latent_propensity_research"
    restart_share = .02
    theta_grid = tuple(1 / (1 + exp(-i / 2)) for i in range(-32, 33))
    # Uniform log-odds spacing: Beta density times theta*(1-theta) Jacobian.
    _prior_mass = tuple(exp(1.2 * log(t) + 2.8 * log(1 - t)) for t in theta_grid)
    prior_weights = _normalized(_prior_mass)

    def __init__(self, seed=None):
        super().__init__(seed)
        self._latent_weights = {phase: list(self.prior_weights)
                                for phase in ("opening", "continuation")}
        self._latent_updates = {phase: 0 for phase in self._latent_weights}
        self._latent_forecasts = {}

    @staticmethod
    def _phase(action):
        return "opening" if _field(action, "pile_size_before", 0) == 0 else "continuation"

    @classmethod
    def _label_probabilities(cls, ability):
        if not isfinite(ability) or not 0 <= ability <= 1:
            raise ValueError("invalid claim ability")
        return tuple(theta / (theta + (1 - theta) * ability)
                     for theta in cls.theta_grid)

    def _posterior_bluff(self, hand, state, action):
        _, ability = super()._posterior_bluff(hand, state, action)
        phase = self._phase(action)
        probabilities = self._label_probabilities(ability)
        weights = tuple(self._latent_weights[phase])
        index = _field(action, "action_index")
        if type(index) is int and index >= 0:
            # Repeated scoring of one pending claim must use PRE-label weights.
            phase, probabilities, weights = self._latent_forecasts.setdefault(
                index, (phase, probabilities, weights))
        if _can_prove_false(hand, state, action):
            return 1.0, ability
        return sum(w * p for w, p in zip(weights, probabilities)), ability

    def observe_action(self, action, opponent_hand_size):
        previous = self._counts(self._bluff_evidence)
        super().observe_action(action, opponent_hand_size)
        index = _field(action, "action_index")
        # Base learner gates voluntary/public labels and detached-copy dedup.
        if self._counts(self._bluff_evidence) == previous:
            if (_field(action, "player") != self.player_id
                    and _field(action, "response_was_forced") is not None):
                self._latent_forecasts.pop(index, None)
            return
        forecast = self._latent_forecasts.pop(index, None)
        if forecast is None:
            return  # Never construct a pre-reveal forecast after seeing truth.
        phase, probabilities, weights = forecast
        label = _field(action, "was_bluff")
        posterior = [w * (p if label else 1 - p)
                     for w, p in zip(weights, probabilities)]
        total = sum(posterior)
        if total <= 0:
            raise ValueError("revealed label has zero latent-model support")
        self._latent_weights[phase] = [
            (1 - self.restart_share) * w / total + self.restart_share * prior
            for w, prior in zip(posterior, self.prior_weights)]
        self._latent_updates[phase] += 1

    def reset(self):
        super().reset()
        self._latent_forecasts.clear()

    def reset_profile(self):
        super().reset_profile()
        self._latent_weights = {phase: list(self.prior_weights)
                                for phase in ("opening", "continuation")}
        self._latent_updates = {phase: 0 for phase in self._latent_weights}
        self._latent_forecasts.clear()

    def to_dict(self):
        return {**super().to_dict(), "latent_model_version": 1,
                "latent_weights": {k: list(v) for k, v in self._latent_weights.items()},
                "latent_updates": dict(self._latent_updates)}

    def restore_opponent_profile(self, data):
        candidate = RoundAdaptiveBot()
        if not isinstance(data, dict) or data.get("policy") != self.policy_name:
            raise ValueError("unsupported latent profile")
        candidate.restore_opponent_profile({**data, "policy": candidate.policy_name})
        raw, updates = data.get("latent_weights"), data.get("latent_updates")
        phases = {"opening", "continuation"}
        if (type(data.get("latent_model_version")) is not int
                or data["latent_model_version"] != 1
                or not isinstance(raw, dict) or set(raw) != phases
                or not isinstance(updates, dict) or set(updates) != phases):
            raise ValueError("malformed latent profile")
        for phase in phases:
            weights, count = raw[phase], updates[phase]
            if (not isinstance(weights, list) or len(weights) != len(self.theta_grid)
                    or any(type(w) not in (int, float) or not isfinite(w)
                           or not 0 <= w <= 1 for w in weights)
                    or abs(sum(weights) - 1) > 1e-9
                    or type(count) is not int or count < 0):
                raise ValueError("malformed latent posterior")
        if sum(updates.values()) > sum(self._counts(candidate._bluff_evidence)):
            raise ValueError("latent updates exceed observed labels")
        self._call_evidence = candidate._call_evidence
        self._bluff_evidence = candidate._bluff_evidence
        self._latent_weights = {k: list(v) for k, v in raw.items()}
        self._latent_updates = dict(updates)
        self._latent_forecasts.clear()


class RoundDirectRateBot(RoundAdaptiveBot):
    """Ablate the ability likelihood from the learned revealed-bluff rate.

    A rate among revealed claims is not a generative prior before card
    feasibility. Using it directly is a diagnostic alternative, not a calibrated
    solution: our own selective challenges still bias which labels we observe.
    Keep certified contradictions exact and all other planner components equal.
    """

    policy_name = "round_direct_rate_research"

    def _posterior_bluff(self, hand, state, action):
        _, ability = super()._posterior_bluff(hand, state, action)
        if _can_prove_false(hand, state, action):
            return 1.0, ability
        # Recover the same contextual rate without interpreting ability as a
        # claim-selection likelihood. No hidden outcome/packet is inspected.
        quantity = _packet_size(action)
        return self._bluff_rate(quantity, int(state.get("pile_size", quantity))), ability


class RoundPhaseRateBot(RoundAdaptiveBot):
    """Isolate opening/continuation revealed rates without mixing their labels.

    Same .30 prior/strength and within-phase smoothing as Adaptive. Forecasts
    use the revealed-claim rate directly, retaining certified contradictions.
    This tests a context split against DirectRate, NOT generative claim-choice
    inference: selective reveals, sparse labels and changing opponents remain
    limitations. Call forecasts and action values are unchanged.
    """

    policy_name = "round_phase_rate_research"

    def __init__(self, seed=None):
        super().__init__(seed)
        self._phase_bluff_evidence = {}

    @staticmethod
    def _phase(action):
        return "opening" if _field(action, "pile_size_before", 0) == 0 else "continuation"

    def _posterior_bluff(self, hand, state, action):
        _, ability = super()._posterior_bluff(hand, state, action)
        if _can_prove_false(hand, state, action):
            return 1.0, ability
        prefix = self._phase(action) + ":"
        evidence = {key[len(prefix):]: value
                    for key, value in self._phase_bluff_evidence.items()
                    if key.startswith(prefix)}
        quantity = _packet_size(action)
        probability = self._smoothed_rate(self.bluff_prior, evidence,
            _context(quantity, int(state.get("pile_size", quantity))))
        return probability, ability

    def observe_action(self, action, opponent_hand_size):
        previous = self._counts(self._bluff_evidence)
        super().observe_action(action, opponent_hand_size)
        # Reuse the base learner's reveal/forced/dedup boundary. A new legal
        # bluff-label count is necessary before reading any opponent label.
        if self._counts(self._bluff_evidence) == previous:
            return
        quantity = _packet_size(action)
        pile = max(0, int(_field(action, "pile_size_before", 0))) + quantity
        key = self._phase(action) + ":" + _context(quantity, pile)
        self._add(self._phase_bluff_evidence, key, _field(action, "was_bluff"))

    def reset_profile(self):
        super().reset_profile()
        self._phase_bluff_evidence.clear()

    def to_dict(self):
        return {**super().to_dict(),
                "phase_bluff_evidence": self._serialize(self._phase_bluff_evidence)}

    def restore_opponent_profile(self, data):
        # Validate both parts before mutating the existing profile.
        candidate = RoundAdaptiveBot()
        if not isinstance(data, dict):
            raise ValueError("malformed phase profile")
        candidate.restore_opponent_profile({**data, "policy": candidate.policy_name})
        if data.get("policy") != self.policy_name:
            raise ValueError("unsupported phase profile")
        raw = data.get("phase_bluff_evidence")
        if not isinstance(raw, dict):
            raise ValueError("malformed phase evidence")
        phases, combined = {}, {}
        for key, counts in raw.items():
            if not isinstance(key, str) or ":" not in key:
                raise ValueError("malformed phase context")
            phase, context = key.split(":", 1)
            if phase not in ("opening", "continuation"):
                raise ValueError("malformed phase context")
            decoded = self._deserialize({context: counts})[context]
            phases[key] = decoded
            total = combined.setdefault(context, [0, 0])
            total[0] += decoded[0]
            total[1] += decoded[1]
        if combined != candidate._bluff_evidence:
            raise ValueError("phase evidence disagrees with aggregate evidence")
        self._call_evidence = candidate._call_evidence
        self._bluff_evidence = candidate._bluff_evidence
        self._phase_bluff_evidence = phases


class RoundHonestyMixtureBot(RoundAdaptiveBot):
    """Online log-loss mixture of never-bluff and existing adaptive forecasts.

    This is a diagnostic predictive mixture, NOT a calibrated generative model
    of human archetypes. Predictions are cached BEFORE legal reveals; only
    voluntary revealed labels score the experts. Uncalled claims stay unknown.
    A .02 fixed-share restart allows recovery after an opponent changes style.
    The original planner, card reasoning and offensive call model are unchanged.
    """

    policy_name = "round_honesty_mixture_research"
    restart_share = 0.02

    def __init__(self, seed=None):
        super().__init__(seed)
        self._honest_weight = 0.5
        self._claim_forecasts = {}

    def reset(self):
        super().reset()
        self._claim_forecasts.clear()

    def reset_profile(self):
        super().reset_profile()
        self._honest_weight = 0.5
        self._claim_forecasts.clear()

    def _posterior_bluff(self, hand, state, action):
        base, ability = super()._posterior_bluff(hand, state, action)
        weight = (1 - self.restart_share) * self._honest_weight + self.restart_share * 0.5
        index = _field(action, "action_index")
        if type(index) is int and index >= 0:
            weight, base = self._claim_forecasts.setdefault(index, (weight, base))
        if _can_prove_false(hand, state, action):
            return 1.0, ability
        return (1 - weight) * base, ability

    def observe_action(self, action, opponent_hand_size):
        super().observe_action(action, opponent_hand_size)
        index = _field(action, "action_index")
        if (_field(action, "player") == self.player_id
                or _field(action, "response_was_forced") is None):
            return
        forecast = self._claim_forecasts.pop(index, None)
        if (forecast is None or _field(action, "response_was_forced") is not False
                or _field(action, "bluff_called") is not True):
            return
        # Never access the label until the packet is publicly revealed.
        label = _field(action, "was_bluff")
        if type(label) is not bool:
            return
        weight, base = forecast
        honest_likelihood = 0.0 if label else 1.0
        adaptive_likelihood = base if label else 1 - base
        denominator = weight * honest_likelihood + (1 - weight) * adaptive_likelihood
        if denominator <= 0:
            raise ValueError("revealed outcome has no support under forecast mixture")
        self._honest_weight = weight * honest_likelihood / denominator

    def choose_action(self, hand, state):
        result = super().choose_action(hand, state)
        self.last_decision_telemetry["honest_expert_weight"] = self._honest_weight
        return result

    def to_dict(self):
        return {**super().to_dict(), "honest_expert_weight": self._honest_weight}

    def restore_opponent_profile(self, data):
        weight = data.get("honest_expert_weight") if isinstance(data, dict) else None
        if (type(weight) not in (int, float) or not isfinite(weight)
                or not 0 <= weight <= 1):
            raise ValueError("malformed honest expert weight")
        super().restore_opponent_profile(data)
        self._honest_weight = float(weight)
        self._claim_forecasts.clear()
