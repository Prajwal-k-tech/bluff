"""Opening-only public response expert with rank-group utility, research only.

Uses the existing fixed-share expert lifecycle and legal public reconstruction.
Structural forecasts are simulated hypotheses, not opponent identities or a
history-conditioned posterior. Active-pile choices retain the local policy.
"""
from bots.round_opening_groups_bot import _OpeningGroupUtility
from bots.round_public_expert_response_bot import RoundPublicExpertResponseBot
from bots.round_normalized_defense_bot import RoundPriorOnlyEmpiricalDefenseBot
from bots.round_truth_response_bot import RoundTruthResponseBot
from bots.round_public_adaptive_response_forecast import forecast_public_adaptive_responses


class _OpeningOnlyExpertScoring:
    def _play_utility(self,hand,state,claimed_rank,quantity,truthful,pile_after,call_probability):
        if state.get('active_rank') is not None or pile_after!=quantity:
            return RoundTruthResponseBot._play_utility(self,hand,state,claimed_rank,quantity,
                                                       truthful,pile_after,call_probability)
        return super()._play_utility(hand,state,claimed_rank,quantity,truthful,pile_after,call_probability)


class RoundOpeningPublicExpertBot(_OpeningGroupUtility,_OpeningOnlyExpertScoring,
                                 RoundPublicExpertResponseBot,RoundPriorOnlyEmpiricalDefenseBot):
    policy_name='round_opening_group_public_expert_research_v1'
    expert_observation_scope='opening_only'

    def __init__(self,seed=None):
        super().__init__(seed)
        self.opening_expert_registrations=0
        self.opening_expert_resolutions=0

    def reset(self):
        super().reset()
        self.opening_expert_registrations=0
        self.opening_expert_resolutions=0

    def reset_profile(self):
        super().reset_profile()
        self.opening_expert_registrations=0
        self.opening_expert_resolutions=0

    def restore_opponent_profile(self,data):
        super().restore_opponent_profile(data)
        self.opening_expert_registrations=0
        self.opening_expert_resolutions=0

    def observe_action(self,action,opponent_hand_size):
        before=self.meta_updates
        super().observe_action(action,opponent_hand_size)
        self.opening_expert_resolutions+=self.meta_updates-before

    def choose_action(self,hand,state):
        if state.get('active_rank') is None and state.get('pile_size')==0:
            index=len(state.get('actions',()))
            already_registered=index in self._expert_pending
            choice=RoundPublicExpertResponseBot.choose_action(self,hand,state)
            if not already_registered and index in self._expert_pending:
                self.opening_expert_registrations+=1
            return choice
        return RoundPriorOnlyEmpiricalDefenseBot.choose_action(self,hand,state)


class RoundPrunedOpeningPublicExpertBot(RoundOpeningPublicExpertBot):
    """Skip opening bluffs that cannot beat a truthful full-rank play.

    A false packet removing at most one rank group scores at most one. A
    truthful full-rank packet always removes one group and scores at least one;
    the planner prefers truth on ties. This applies only to a valid free lead.
    """
    policy_name='round_pruned_opening_group_public_expert_research_v1'

    def prepare_injected_root_forecast(self, hand, state, choice):
        """Record the injected opening, rather than a different planned action."""
        if type(self) is not RoundPrunedOpeningPublicExpertBot:
            raise TypeError('injected-root lifecycle is not defined for this subclass')
        self.observe_player_view(state)
        index = len(state.get('actions', ()))
        registered = index in self._expert_pending
        self._expert_pending.pop(index, None)
        if state.get('active_rank') is not None or state.get('pile_size') != 0:
            return
        seed = self.rng.randrange(2**63)
        if choice.get('action') != 'play' or len(choice['cards']) >= len(hand):
            return
        packet, rank = choice['cards'], choice['rank']
        truth = all(card.rank == rank for card in packet)
        quantity = len(packet)
        _, detail = RoundTruthResponseBot._play_utility(
            self, hand, state, rank, quantity, truth, quantity, 0)
        def forecast(prior):
            return forecast_public_adaptive_responses(
                hand, state, [(list(packet), rank)], seed=seed, samples=self.samples,
                prior_public_profile=prior, reuse_free_lead=True,
                condition_opening=self.condition_opening)[0]['probabilities']['challenge']
        memory = forecast(self._public_prior)
        fresh = memory if self._public_prior is None else forecast(None)
        proof = self._certifiable_probability(hand, state, rank, quantity)
        self._expert_pending[index] = dict(
            probabilities=[detail['call_probability'], max(proof, memory), max(proof, fresh)],
            truthful=truth)
        if not registered:
            self.opening_expert_registrations += 1

    def _play_candidates(self,hand,state):
        opening=state.get('active_rank') is None and state.get('pile_size')==0
        if not opening:
            yield from super()._play_candidates(hand,state)
            return
        groups=len({card.rank for card in hand})
        for packet,rank,truthful in super()._play_candidates(hand,state):
            if not truthful:
                remaining={card.rank for card in hand if card not in packet}
                if groups-len(remaining)<=1:
                    continue
            yield packet,rank,truthful
