"""Opening-only rank-group utility surrogate; not a solved win probability.

At a free lead P=0, caught false packets restore the original own hand; caught
truthful packets can add at most ONE rank group to the opponent. Compare these
expected group changes only among opening plays; never mix units with call/pass.
Opponent rank-absence probability remains the inherited exchangeable surrogate,
not a claim-conditioned hidden-hand posterior.
"""
from collections import Counter
from bots.round_normalized_defense_bot import RoundPriorOnlyEmpiricalDefenseBot
from bots.round_bayesian_pool_response_bot import RoundBayesianPoolResponseBot


class _OpeningGroupUtility:
    def _plan(self,hand,state):
        self._opening_packets={(rank,len(packet),truth):packet for packet,rank,truth
            in self._play_candidates(hand,state)} if state.get('active_rank') is None and state.get('pile_size')==0 else {}
        try:return super()._plan(hand,state)
        finally:del self._opening_packets

    def _play_utility(self,hand,state,claimed_rank,quantity,truthful,pile_after,call_probability):
        value,detail=super()._play_utility(hand,state,claimed_rank,quantity,truthful,pile_after,call_probability)
        if state.get('active_rank') is not None or pile_after!=quantity:return value,detail
        packet=self._opening_packets[claimed_rank,quantity,truthful]
        remaining=[c for c in hand if c not in packet]
        reduction=len({c.rank for c in hand})-len({c.rank for c in remaining})
        p=detail['call_probability']
        absence=1-self._hold_probability(hand,state,claimed_rank,state['opponent_hand_size'],1)
        value=reduction+p*absence if truthful else (1-p)*reduction
        detail.update(utility=value,utility_basis='opening_expected_rank_group_advantage_v1',
            own_rank_group_reduction=reduction,opponent_rank_absence=absence)
        return value,detail


class RoundLocalOpeningGroupsBot(_OpeningGroupUtility,RoundPriorOnlyEmpiricalDefenseBot):
    policy_name='round_local_opening_groups_research_v1'


class RoundPooledOpeningGroupsBot(_OpeningGroupUtility,RoundBayesianPoolResponseBot):
    policy_name='round_pooled_opening_groups_research_v1'
