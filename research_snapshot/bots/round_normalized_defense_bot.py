"""Fresh-opening posterior corrections for controlled research comparisons.

The normalized variant is a declared likelihood-ratio surrogate. It models a
false claim as coming from a uniformly sampled hand and a true claim as
conditioned on eligibility; it is not a general human-opponent posterior.
"""

from bots.round_opening_forecast import _opening_inputs
from bots.round_policy import _can_prove_false, _hypergeom_at_least
from bots.round_public_expert_response_bot import (
    RoundCertifiedOpeningPublicExpertResponseBot,
)
from bots.round_truth_response_bot import RoundTruthResponseBot


class _FreshOpeningDefenseBot(RoundCertifiedOpeningPublicExpertResponseBot):
    """Common guarded first-opening scope; all other behavior is inherited."""

    def _supported_fresh_opening(self, hand, state, action):
        if not isinstance(state, dict) or len(hand) != 14:
            return False
        if (state.get("round_number") != 1 or state.get("turn_number") != 1
                or state.get("hand_size") != 14 or state.get("draw_pile_size") != 24
                or state.get("private_pickups") != []
                or state.get("own_pile_cards") != []):
            return False
        actions = state.get("actions")
        if not isinstance(actions, (tuple, list)) or len(actions) != 1:
            return False
        try:
            _rank, quantity, _prestate = _opening_inputs(hand, state)
        except (TypeError, ValueError):
            return False
        pending = state.get("last_action")
        return (pending is action and quantity >= 1
                and state.get("opponent_hand_size", -1) + quantity == 14)

    def _opening_components(self, hand, state, action):
        # Keep the parent's eligibility calculation; discard its posterior.
        # Exact physical disproof is handled before this call by each override.
        return super()._posterior_bluff(hand, state, action)


class RoundPriorOnlyDefenseBot(_FreshOpeningDefenseBot):
    """Use the empirical bluff prior directly on a supported fresh opening."""

    policy_name = "round_prior_only_defense_research_v1"

    def _posterior_bluff(self, hand, state, action):
        if not self._supported_fresh_opening(hand, state, action):
            return super()._posterior_bluff(hand, state, action)
        if _can_prove_false(hand, state, action):
            return 1.0, 0.0
        _base, can_truthfully_hold = self._opening_components(hand, state, action)
        quantity = len(action.cards_played)
        prior = min(1.0, max(0.0, self._bluff_rate(quantity, state["pile_size"])))
        return prior, can_truthfully_hold


class RoundNormalizedDefenseBot(_FreshOpeningDefenseBot):
    """Apply the declared eligibility likelihood-ratio surrogate at opening."""

    policy_name = "round_normalized_defense_research_v1"

    def _posterior_bluff(self, hand, state, action):
        if not self._supported_fresh_opening(hand, state, action):
            return super()._posterior_bluff(hand, state, action)
        if _can_prove_false(hand, state, action):
            return 1.0, 0.0
        _base, eligibility = self._opening_components(hand, state, action)
        quantity = len(action.cards_played)
        prior = min(1.0, max(0.0, self._bluff_rate(quantity, state["pile_size"])))
        if prior == 0.0 or prior == 1.0:
            return prior, eligibility
        reference = _hypergeom_at_least(52, 4, 14, quantity)
        ratio = eligibility / reference if reference > 0.0 else 0.0
        denominator = prior + (1.0 - prior) * ratio
        posterior = 1.0 if denominator <= 0.0 else prior / denominator
        return min(1.0, max(0.0, posterior)), eligibility


class RoundPriorOnlyEmpiricalDefenseBot(RoundTruthResponseBot):
    """Same opening-defense control on the cheaper empirical response policy."""

    policy_name = 'round_prior_only_empirical_defense_research_v1'

    def _posterior_bluff(self, hand, state, action):
        if not _FreshOpeningDefenseBot._supported_fresh_opening(self, hand, state, action):
            return super()._posterior_bluff(hand, state, action)
        if _can_prove_false(hand, state, action):
            return 1.0, 0.0
        _base, eligibility = super()._posterior_bluff(hand, state, action)
        prior = self._bluff_rate(len(action.cards_played), state['pile_size'])
        return min(1.0, max(0.0, prior)), eligibility
