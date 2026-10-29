"""Research-only adaptive bot with a conditional center-count risk signal.

The Honest response-mixture weight below is a heuristic policy-response weight,
not a calibrated identity posterior. Center counts are public claim quantities;
this policy never inspects unrevealed packet identities.
"""
from copy import copy, deepcopy
import random

from bots.round_honest_hypothesis_memory import honest_count_hypothesis_bounds
from bots.round_pickup_adaptive_bot import RoundPickupAdaptiveBot
from bots.round_policy_response_forecast import forecast_policy_responses
from bots.round_policy_response_learner import ConsistentPolicyResponseLearner


class RoundCenterPolicyBot(RoundPickupAdaptiveBot):
    policy_name = "round_center_policy_research_v1"
    response_forecast_samples = 4
    register_injected_root_forecast = True
    requires_complete_private_history_for_search = True

    def __init__(self, seed=None):
        super().__init__(seed)
        self.policy_learner = ConsistentPolicyResponseLearner()

    def reset(self):
        super().reset()
        if hasattr(self, "policy_learner"):
            self.policy_learner.begin_deal()

    def reset_profile(self):
        super().reset_profile()
        self.policy_learner = ConsistentPolicyResponseLearner()

    def _consume_policy_responses(self, state):
        viewer = state.get("player_id")
        for action in state.get("actions", ()):
            if action.player == viewer and action.response_kind is not None:
                self.policy_learner.observe(
                    action.action_index, action.response_kind,
                    forced=action.response_was_forced)

    def observe_player_view(self, state):
        super().observe_player_view(state)
        self._consume_policy_responses(state)

    def _play_utility(self, hand, state, claimed_rank, quantity, truthful,
                      pile_after, call_probability):
        utility, detail = super()._play_utility(
            hand, state, claimed_rank, quantity, truthful, pile_after,
            call_probability)
        if getattr(self, "_center_floor_disabled", False):
            return utility, detail
        active = state.get("active_rank")
        center_total = 0
        if not truthful and active is not None:
            viewer = state.get("player_id")
            round_number = state.get("round_number")
            for action in state.get("actions", ()):
                if (action.player != viewer
                        and action.round_number == round_number
                        and action.claimed_rank == claimed_rank):
                    center_total += len(action.cards_played)

        center_condition = (not truthful and active is not None
                            and center_total + quantity > 4)
        compatible = (honest_count_hypothesis_bounds(state)["compatible"] is True
                      if center_condition else False)
        activated = center_condition and compatible
        detail = dict(detail)
        detail["call_probability"] = call_probability
        detail["center_honest_floor"] = {
            "activated": activated,
            "opponent_claimed_rank_count": center_total,
            "method": "heuristic_policy_response_weight_not_identity_posterior",
        }
        if activated:
            honest_weight = self.policy_learner.weights["Honest"]
            floor = .98 * honest_weight + .005
            if floor > call_probability:
                self._center_candidates_raised = getattr(self,"_center_candidates_raised",0)+1
            call_probability = max(call_probability, floor)
            utility, parent_detail = super()._play_utility(
                hand, state, claimed_rank, quantity, truthful, pile_after,
                call_probability)
            detail.update(parent_detail)
            detail["call_probability"] = call_probability
        return utility, detail

    def _plan(self, hand, state):
        # Same-state ablation changes only the center floor, not observations,
        # RNG, profile or candidate ordering. No forecast/search is executed.
        shadow = copy(self)
        shadow._center_floor_disabled = True
        baseline, _ = RoundPickupAdaptiveBot._plan(shadow, hand, state)
        self._center_candidates_raised = 0
        selected, detail = super()._plan(hand, state)
        detail["center_candidates_raised"] = self._center_candidates_raised
        detail["center_floor_action_changed"] = selected != baseline
        return selected, detail

    def choose_action(self, hand, state):
        choice = super().choose_action(hand, state)
        self._forecast_selected_play(hand, state, choice)
        return choice

    def prepare_injected_root_forecast(self, hand, state, choice):
        """Apply the normal pre-decision observation before an injected move."""
        self.observe_player_view(state)
        self.policy_learner.discard_forecast(len(state.get("actions", ())))
        self._forecast_selected_play(hand, state, choice)

    def _forecast_selected_play(self, hand, state, choice):
        """Register only the move actually chosen, including simulated branches."""
        if (choice.get("action") == "play"
                and len(choice["cards"]) < len(hand)):
            probe = random.Random()
            probe.setstate(self.rng.getstate())
            result = forecast_policy_responses(
                hand, state, choice["cards"], choice["rank"],
                seed=probe.randrange(2**63), samples=self.response_forecast_samples,
                hypothesis_consistent=True)
            if not result["forced"]:
                self.policy_learner.forecast(
                    len(state.get("actions", ())), result["probabilities"])

    def to_dict(self):
        return {**super().to_dict(),
                "center_policy_learner": self.policy_learner.to_dict()}

    def restore_opponent_profile(self, data):
        expected = set(self.to_dict())
        if (not isinstance(data, dict) or set(data) != expected
                or data.get("policy") != self.policy_name):
            raise ValueError("unsupported center-policy profile")
        parent = deepcopy(data)
        learner_profile = parent.pop("center_policy_learner")
        parent["policy"] = RoundPickupAdaptiveBot.policy_name
        staged = RoundPickupAdaptiveBot()
        staged.restore_opponent_profile(parent)
        learner = ConsistentPolicyResponseLearner()
        learner.restore(deepcopy(learner_profile))
        rng = self.rng
        self.__dict__.update(staged.__dict__)
        self.rng = rng
        self.policy_name = type(self).policy_name
        self.policy_learner = learner


class RoundResolvedResponsePolicyBot(RoundCenterPolicyBot):
    """Research ablation: only increase response forecast resolution to 32."""
    policy_name = "round_center_response32_research_v1"
    response_forecast_samples = 32
