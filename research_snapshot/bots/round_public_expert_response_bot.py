"""Unregistered response-expert mixture; family hypotheses are not identities.

Empirical own-truth rates compete with public-Adaptive memory/fresh hypotheses.
Weights learn only from frozen forecasts of selected voluntary plays. All use
immediate card-balance utility, not a validated win-value function.
"""
from copy import deepcopy
from math import isfinite
from bots.round_policy import RoundAdaptiveBot
from bots.round_truth_response_bot import RoundTruthResponseBot
from bots.round_public_opponent_model import public_adaptive_from_history
from bots.round_public_adaptive_response_forecast import forecast_public_adaptive_responses


class RoundPublicExpertResponseBot(RoundTruthResponseBot):
    policy_name='round_public_expert_response_research_v1'
    samples=4
    share=.05
    condition_opening=False

    def __init__(self,seed=None):
        super().__init__(seed)
        self.expert_weights=[1/3]*3;self.meta_updates=0
        self.public_latest=None;self._public_prior=None
        self._expert_pending={};self._forecast_cache={};self._decision_view=None
        self._conditioned_opening_batches=0

    def reset(self):
        super().reset()
        self._public_prior=deepcopy(self.public_latest)
        self._expert_pending={};self._forecast_cache={};self._decision_view=None
        self._conditioned_opening_batches=0

    def reset_profile(self):
        super().reset_profile()
        self.expert_weights=[1/3]*3;self.meta_updates=0
        self.public_latest=None;self._public_prior=None
        self._expert_pending={};self._forecast_cache={};self._decision_view=None
        self._conditioned_opening_batches=0

    def observe_player_view(self,state):
        super().observe_player_view(state)
        actor=public_adaptive_from_history(state['actions'],opponent_id=1-self.player_id,
            prior_public_profile=self._public_prior)
        self.public_latest=actor.to_dict()

    def observe_action(self,action,opponent_hand_size):
        index=getattr(action,'action_index',None)
        pending=self._expert_pending.get(index)
        if (pending is not None and self.player_id is not None
            and action.player==self.player_id
            and getattr(action,'response_player',None)==1-self.player_id
            and getattr(action,'response_was_forced',None) is False
            and type(getattr(action,'response_was_call',None)) is bool
            and type(getattr(action,'was_bluff',None)) is bool
            and action.was_bluff is not pending['truthful']):
            probabilities=self._expert_pending.pop(index)['probabilities']
            called=action.response_was_call
            likelihood=[max(1e-12,p if called else 1-p) for p in probabilities]
            mass=[w*p for w,p in zip(self.expert_weights,likelihood)]
            total=sum(mass)
            self.expert_weights=[(1-self.share)*m/total+self.share/3 for m in mass]
            self.meta_updates+=1
        super().observe_action(action,opponent_hand_size)

    def choose_action(self,hand,state):
        self._forecast_cache={};self._decision_view=deepcopy(state)
        self._decision_seed=self.rng.randrange(2**63)
        try:
            # Batch all base representatives without rank pruning. Each model
            # reuses its sampled worlds and public actor across rank/quantity.
            active=state.get('active_rank')
            guaranteed=(hand and len(hand)<=4 and len({c.rank for c in hand})==1
                        and (active is None or hand[0].rank==active))
            packets={}
            if not guaranteed and state.get('can_play',bool(hand)):
                for packet,rank,truth in self._play_candidates(hand,state):
                    if len(packet)<len(hand):packets.setdefault((rank,len(packet)),list(packet))
            if packets:
                keys=list(packets)
                choices=[(packets[key],key[0]) for key in keys]
                memory=forecast_public_adaptive_responses(hand,self._decision_view,choices,
                    seed=self._decision_seed,samples=self.samples,prior_public_profile=self._public_prior,
                    reuse_free_lead=True,condition_opening=self.condition_opening)
                if (memory and memory[0].get('scope','').startswith(
                        'certified fresh-opening-conditioned')):
                    self._conditioned_opening_batches+=1
                # Before prior-session evidence exists, these experts have
                # identical inputs/seeds. Reuse output, not their weights.
                fresh=(memory if self._public_prior is None else
                    forecast_public_adaptive_responses(hand,self._decision_view,choices,
                        seed=self._decision_seed,samples=self.samples,prior_public_profile=None,
                        reuse_free_lead=True,condition_opening=self.condition_opening))
                if (fresh is not memory and fresh
                    and fresh[0].get('scope','').startswith(
                        'certified fresh-opening-conditioned')):
                    self._conditioned_opening_batches+=1
                forecasts=[memory,fresh]
                self._forecast_cache={key:tuple(batch[i]['probabilities']['challenge'] for batch in forecasts)
                    for i,key in enumerate(keys)}
            choice=super().choose_action(hand,state)
            if choice['action']=='play' and len(choice['cards'])<len(hand):
                self._expert_pending[len(state['actions'])]=dict(
                    probabilities=list(self.last_decision_telemetry['expert_probabilities']),
                    truthful=all(c.rank==choice['rank'] for c in choice['cards']))
            self.last_decision_telemetry.update(expert_weights=list(self.expert_weights),
                model_pair_queries=len(self._forecast_cache),samples_per_model=self.samples)
            return choice
        finally:
            self._decision_view=None

    def _play_utility(self,hand,state,claimed_rank,quantity,truthful,pile_after,call_probability):
        value,detail=super()._play_utility(hand,state,claimed_rank,quantity,truthful,pile_after,call_probability)
        if quantity==len(hand):return value,detail
        key=(claimed_rank,quantity)
        if key not in self._forecast_cache:raise RuntimeError('missing batched legal candidate forecast')
        proof=self._certifiable_probability(hand,state,claimed_rank,quantity)
        probabilities=[detail['call_probability']]+[max(proof,p) for p in self._forecast_cache[key]]
        p=sum(w*v for w,v in zip(self.expert_weights,probabilities))
        return (quantity+p*pile_after if truthful else quantity-p*pile_after),dict(
            call_probability=p,expert_probabilities=probabilities,
            response_model='empirical_public_memory_fresh_fixed_share_v1')

    def to_dict(self):
        parent=super().to_dict();parent['policy']=RoundTruthResponseBot.policy_name
        return dict(version=1,policy=self.policy_name,parent=parent,
            public_profile=deepcopy(self.public_latest),weights=list(self.expert_weights),meta_updates=self.meta_updates)

    def restore_opponent_profile(self,data):
        if (not isinstance(data,dict) or set(data)!={'version','policy','parent','public_profile','weights','meta_updates'}
            or type(data['version']) is not int or data['version']!=1 or data['policy']!=self.policy_name):
            raise ValueError('unsupported expert response profile')
        weights=data['weights'];updates=data['meta_updates']
        if (not isinstance(weights,list) or len(weights)!=3
            or any(type(w) not in (int,float) or not isfinite(w) or w<=0 for w in weights)
            or abs(sum(weights)-1)>1e-9 or type(updates) is not int or updates<0):
            raise ValueError('malformed response experts')
        parent=RoundTruthResponseBot(0);parent.restore_opponent_profile(deepcopy(data['parent']))
        public=deepcopy(data['public_profile'])
        if public is not None:
            staged=RoundAdaptiveBot(0);staged.restore_opponent_profile(public)
            if staged.to_dict()!=public:raise ValueError('noncanonical public profile')
        self._call_evidence=deepcopy(parent._call_evidence);self._bluff_evidence=deepcopy(parent._bluff_evidence)
        self.truth_calls=deepcopy(parent.truth_calls);self.expert_weights=list(weights);self.meta_updates=updates
        self.public_latest=public;self._public_prior=deepcopy(public)
        self._truth_resolved.clear();self._resolved_action_objects.clear()
        self._expert_pending={};self._forecast_cache={};self._decision_view=None
        self._conditioned_opening_batches=0


class RoundCertifiedOpeningPublicExpertResponseBot(RoundPublicExpertResponseBot):
    """Opt-in research variant for supported fresh Adaptive openings only."""
    policy_name='round_certified_opening_public_expert_response_research_v1'
    condition_opening=True
