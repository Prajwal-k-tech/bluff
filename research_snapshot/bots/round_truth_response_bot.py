"""Unregistered own-packet-truth conditioned call-rate research candidate.

Own truth is legally known, not inferred opponent truth. Fixed Beta prior per
quantity/pile/truth context avoids reusing observations as a learned prior.
"""
from copy import deepcopy
from bots.round_policy import RoundAdaptiveBot,_context


class RoundTruthResponseBot(RoundAdaptiveBot):
    policy_name='round_own_truth_response_research_v1'

    def __init__(self,seed=None):
        super().__init__(seed);self.truth_calls={'truth':{},'bluff':{}};self._truth_resolved=set()

    def reset(self):
        super().reset();self._truth_resolved.clear()

    def reset_profile(self):
        super().reset_profile();self.truth_calls={'truth':{},'bluff':{}};self._truth_resolved.clear()

    def observe_player_view(self,state):
        super().observe_player_view(state)
        for action in state.get('actions',()):self.observe_action(action,state['opponent_hand_size'])

    def observe_action(self,action,opponent_hand_size):
        super().observe_action(action,opponent_hand_size)
        index=getattr(action,'action_index',None);truth=getattr(action,'was_bluff',None)
        called=getattr(action,'response_was_call',None)
        cards=getattr(action,'cards_played',())
        if (type(index) is not int or index<0 or index in self._truth_resolved
            or self.player_id is None or action.player!=self.player_id
            or type(truth) is not bool or type(called) is not bool
            or getattr(action,'response_was_forced',None) is not False
            or getattr(action,'response_player',None)!=1-self.player_id
            or not 1<=len(cards)<=4):return
        key=_context(len(cards),action.pile_size_before+len(cards))
        self._add(self.truth_calls['bluff' if truth else 'truth'],key,called)
        self._truth_resolved.add(index)

    def _response_counts(self,truthful,key):
        return self.truth_calls['truth' if truthful else 'bluff'].get(key,[0,0])

    def _play_utility(self,hand,state,claimed_rank,quantity,truthful,pile_after,call_probability):
        key=_context(quantity,pile_after)
        calls,declines=self._response_counts(truthful,key)
        learned=(1.4+calls)/(4+calls+declines)
        proof=self._certifiable_probability(hand,state,claimed_rank,quantity)
        probability=1. if quantity==len(hand) else max(learned,proof)
        utility=quantity+probability*pile_after if truthful else quantity-probability*pile_after
        return utility,dict(call_probability=probability,truth_response_calls=calls,
            truth_response_declines=declines,response_model='fixed_beta_truth_quantity_pile_v1')

    def to_dict(self):
        parent=super().to_dict();parent['policy']=RoundAdaptiveBot.policy_name
        return dict(version=1,policy=self.policy_name,parent=parent,
            truth_calls={name:self._serialize(values) for name,values in self.truth_calls.items()})

    def restore_opponent_profile(self,data):
        if (not isinstance(data,dict) or set(data)!={'version','policy','parent','truth_calls'}
            or type(data['version']) is not int or data['version']!=1 or data['policy']!=self.policy_name
            or not isinstance(data['truth_calls'],dict) or set(data['truth_calls'])!={'truth','bluff'}):
            raise ValueError('unsupported truth response profile')
        parent=RoundAdaptiveBot(0);parent.restore_opponent_profile(deepcopy(data['parent']))
        counts={name:self._deserialize(values) for name,values in data['truth_calls'].items()}
        self._call_evidence=deepcopy(parent._call_evidence);self._bluff_evidence=deepcopy(parent._bluff_evidence)
        self.truth_calls=counts;self._truth_resolved.clear();self._resolved_action_objects.clear()


class RoundPooledBetaResponseControl(RoundTruthResponseBot):
    """Same fixed Beta/context implementation, pooled truth labels control."""
    policy_name='round_pooled_beta_response_control_v1'

    def _response_counts(self,truthful,key):
        a=self.truth_calls['truth'].get(key,[0,0]);b=self.truth_calls['bluff'].get(key,[0,0])
        return [a[0]+b[0],a[1]+b[1]]


class RoundPartialPoolResponseBot(RoundTruthResponseBot):
    """Research power-likelihood estimate sharing 1/4 weight across contexts.

    Same-truth observations only; queried-cell counts appear exactly once.
    This fixed shrinkage rule is not a calibrated posterior under context
    heterogeneity, and does not itself address opponent strategy changes.
    """
    policy_name='round_partial_pool_truth_response_research_v1'

    def _response_counts(self,truthful,key):
        evidence=self.truth_calls['truth' if truthful else 'bluff']
        local=evidence.get(key,[0,0])
        other=[sum(value[i] for context,value in evidence.items() if context!=key) for i in (0,1)]
        return [local[i]+.25*other[i] for i in (0,1)]

    def _play_utility(self,*args,**kwargs):
        utility,detail=super()._play_utility(*args,**kwargs)
        detail['response_model']='same_truth_cross_context_power_likelihood_quarter_v1'
        return utility,detail
