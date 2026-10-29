"""Research conditional worlds for a declared Adaptive opening-law regime.

Not a human model or a general history posterior. The regime's opening is its
largest truthful group, lowest-rank tie, so its observed event gives exact
rank-count constraints on a fresh uniform deal. Unsupported inputs fail closed.
"""
from random import Random
from bots.round_determinization import sample_root_world
from bots.round_opening_forecast import _opening_inputs
from bots.round_policy import RoundAdaptiveBot
from cards import Rank


def sample_certified_opening_world(hand,view,rng,*,prior_public_profile=None):
    if (not isinstance(view,dict) or set(view).intersection(
            {'hands','opponent_hand','game','truth','target','opponent_profile'})
        or view.get('own_hand')!=list(hand) or not isinstance(rng,Random)):
        raise ValueError('legal focal hand/view and owned RNG required')
    rank,q,before=_opening_inputs(hand,view)
    if (len(hand)!=14 or len(view['actions'])!=1 or view['round_number']!=1
        or view.get('turn_number')!=1 or before['opponent_hand_size']!=14
        or before['draw_pile_size']!=24 or view.get('private_pickups')
        or view.get('own_pile_cards') or q<2):
        raise ValueError('unsupported first-deal opening domain')
    if any(a.was_bluff is not None or any(c is not None for c in a.cards_played)
           for a in (view['last_action'],view['actions'][0])):
        raise ValueError('opponent opening identities must remain hidden')
    model=RoundAdaptiveBot(0)
    if prior_public_profile is not None:
        model.restore_opponent_profile(prior_public_profile)
        if model.to_dict()!=prior_public_profile:raise ValueError('noncanonical public checkpoint')
    if any(model._call_rate(n,n)<.35 for n in (1,2,3,4)):
        raise ValueError('unsupported opening-law call-rate regime')
    # A pair exists in14cards; truthfulpair EV>=2.7 > any bluff EV<=2.6.
    # q3/4 share a call context; q3 EV>=4.05 > any pair EV<=4, so the
    # largest group wins, with the canonical sorted-hand lowest-rank tie.
    caps={r:q-1 if int(r)<int(rank) else q for r in Rank}
    return sample_root_world(hand,view,rng,pending_truth=True,
        fixed_active_counts=(0,q),preclaim_rank_caps=caps)
