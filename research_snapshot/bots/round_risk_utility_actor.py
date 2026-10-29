"""Unregistered risk-floor and tremble actor for corrected Bluff, v1."""

from copy import deepcopy
from math import comb, perm
from random import Random

from cards import Rank
from bots.round_blended_response_actor import _validate_view
from bots.round_pickup_adaptive_bot import RoundPickupAdaptiveBot
from bots.round_policy import RoundRandomBot
from game_v2 import RULESET_ID


class RoundRiskUtilityActor(RoundPickupAdaptiveBot):
    """PickupAdaptive chooser with a minimum perceived challenge rate.

    Tremble mixes its deterministic best action with the existing hierarchical
    Random policy. This is a research actor, not a registered playing policy.
    """

    policy_name = "round_risk_utility_actor_v1"
    model_id = "round_risk_utility_actor_v1"
    version = 1

    def __init__(self, seed=None, *, risk_floor, tremble=0.05):
        super().__init__(seed)
        self.risk_floor = self._probability(risk_floor, "risk_floor")
        self.tremble = self._probability(tremble, "tremble")

    @staticmethod
    def _probability(value, name):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be a probability in [0, 1]")
        return float(value)

    def _call_rate(self, quantity, pile_size):
        return max(self.risk_floor, super()._call_rate(quantity, pile_size))

    @classmethod
    def from_inferred_actor(cls, actor, seed=None, *, risk_floor, tremble=.05):
        """Copy legally reconstructed CURRENT-deal evidence AND consumed IDs.

        Caller supplies its own history reconstruction, never a hidden runtime
        opponent. Profile restore is instead for a PRE-deal persistent seed.
        """
        if type(actor) is not RoundPickupAdaptiveBot:
            raise ValueError('canonical legally inferred Pickup actor required')
        result = cls(seed, risk_floor=risk_floor, tremble=tremble)
        for field in ('_call_evidence', '_bluff_evidence', '_resolved_action_objects'):
            setattr(result, field, deepcopy(getattr(actor, field)))
        result.player_id = actor.player_id
        return result

    @staticmethod
    def _event(action, hand):
        family = "challenge" if action["action"] == "call_bluff" else action["action"]
        if family != "play":
            return family, None, 0, None
        rank = action["rank"]
        cards = action["cards"]
        return family, rank, len(cards), all(card.rank == rank for card in cards)

    def _random_event_distribution(self, hand, view):
        random_actor = RoundRandomBot(seed=0)
        random_actor.player_id = view["player_id"]
        families = random_actor._legal_families(list(hand), view)
        law = {}
        family_mass = 1.0 / len(families)
        for family in families:
            if family != "play":
                key = ("challenge" if family == "call_bluff" else family,
                       None, 0, None)
                law[key] = law.get(key, 0.0) + family_mass
                continue
            n = len(hand)
            quantities = range(1, min(4, n) + 1)
            quantity_mass = family_mass / len(quantities)
            active = view.get("active_rank")
            ranks = (active,) if active is not None else tuple(Rank)
            for quantity in quantities:
                subset_count = comb(n, quantity)
                for rank in ranks:
                    matching = sum(card.rank == rank for card in hand)
                    truth_probability = (comb(matching, quantity) / subset_count
                                         if matching >= quantity else 0.0)
                    for truth, probability in ((True, truth_probability),
                                               (False, 1.0-truth_probability)):
                        if probability:
                            key = ("play", rank, quantity, truth)
                            law[key] = law.get(key, 0.0) + (
                                quantity_mass / len(ranks) * probability)
        return law

    def event_distribution(self, hand, view):
        """Return the coarse actor event law without changing actor state."""
        _validate_view(hand, view)
        detached = deepcopy(self)
        detached.player_id = view["player_id"]
        detached_view = deepcopy(view)
        detached.observe_player_view(detached_view)
        action, _ = detached._plan(list(hand), detached_view)
        deterministic = self._event(action, hand)
        if self.tremble == 0.0:
            return {deterministic: 1.0}
        random_law = self._random_event_distribution(hand, view)
        result = {key: self.tremble * probability
                  for key, probability in random_law.items()}
        result[deterministic] = result.get(deterministic, 0.0) + 1.0-self.tremble
        return result

    def sample_conditional_public_action(self, hand, view, public_event, rng):
        """Sample this actor's physical move conditional on a public event.

        Research world propagation only. The event contains family/rank/count,
        NEVER the unobserved packet's truth or identities. This is the exact
        deterministic/tremble mixture, not a new acting strategy.
        """
        if not isinstance(rng, Random):
            raise ValueError('caller-owned Random required')
        action, deterministic_mass, random_mass = self._conditional_components(hand, view, public_event)
        evidence = deterministic_mass + random_mass
        if rng.random() < deterministic_mass/evidence:
            choice = deepcopy(action)
        else:
            family, rank, quantity = public_event
            choice = ({'action': 'play', 'rank': rank,
                       'cards': rng.sample(list(hand), quantity)} if family == 'play'
                      else {'action': 'call_bluff' if family == 'challenge' else family})
        return dict(choice=choice, public_event_probability=evidence,
                    deterministic_component_probability=deterministic_mass/evidence)

    def _conditional_components(self, hand, view, public_event):
        from bots.round_public_risk_filter import PublicRiskFilter
        PublicRiskFilter._event(public_event)
        _validate_view(hand, view)
        detached = deepcopy(self)
        detached.player_id = view['player_id']
        detached_view = deepcopy(view)
        detached.observe_player_view(detached_view)
        action, _ = detached._plan(list(hand), detached_view)
        deterministic_mass = ((1-self.tremble)
            if self._event(action, hand)[:3] == public_event else 0.)
        random_mass = self.tremble * sum(p for event, p in
            self._random_event_distribution(hand, view).items() if event[:3] == public_event)
        evidence = deterministic_mass + random_mass
        if evidence <= 0:
            raise ValueError('public event has zero actor support')
        return action, deterministic_mass, random_mass

    @staticmethod
    def _sample_false_packet(hand,rank,quantity,rng):
        """Uniform ordered packet conditional on at least one false card."""
        matching=[card for card in hand if card.rank==rank]
        other=[card for card in hand if card.rank!=rank]
        strata=[(k,comb(len(matching),k)*comb(len(other),quantity-k))
                for k in range(quantity) if k<=len(matching) and quantity-k<=len(other)]
        total=sum(count for _,count in strata)
        if not total:
            raise ValueError('no false packet support')
        draw=rng.randrange(total)
        for k,count in strata:
            if draw<count:
                cards=rng.sample(matching,k)+rng.sample(other,quantity-k)
                rng.shuffle(cards)
                return cards
            draw-=count
        raise RuntimeError('false packet categorical accounting failed')

    def stratified_conditional_public_actions(self,hand,view,public_event,rng):
        """Weighted research scenarios, not independent posterior samples.

        Preserve truth strata analytically; identities inside each random
        stratum are sampled uniformly. Deterministic/random overlap is a
        legitimate mixture, not a reason to discard either component's mass.
        """
        if not isinstance(rng,Random):
            raise ValueError('caller-owned Random required')
        action,deterministic_mass,random_mass=self._conditional_components(hand,view,public_event)
        evidence=deterministic_mass+random_mass
        family,rank,quantity=public_event
        if family!='play':
            return dict(public_event_probability=evidence,branches=[dict(component='nonplay',
                choice=dict(action='call_bluff' if family=='challenge' else family),weight=1.,truthful=None)])
        branches=[]
        if deterministic_mass:
            branches.append(dict(component='deterministic',choice=deepcopy(action),
                weight=deterministic_mass/evidence,truthful=self._event(action,hand)[3]))
        matching=[card for card in hand if card.rank==rank]
        truth_fraction=(comb(len(matching),quantity)/comb(len(hand),quantity)
                        if len(matching)>=quantity else 0.)
        for truthful,fraction in ((True,truth_fraction),(False,1-truth_fraction)):
            if not random_mass or not fraction:
                continue
            cards=(rng.sample(matching,quantity) if truthful else
                   self._sample_false_packet(hand,rank,quantity,rng))
            branches.append(dict(component='random_truth' if truthful else 'random_bluff',
                choice=dict(action='play',rank=rank,cards=cards),
                weight=random_mass*fraction/evidence,truthful=truthful))
        return dict(public_event_probability=evidence,branches=branches)

    def conditional_public_action_probability(self, hand, view, public_event, choice):
        """Likelihood of an exact ordered packet after it is legally revealed."""
        action, deterministic_mass, random_mass = self._conditional_components(hand, view, public_event)
        return self._physical_probability(hand, public_event, choice, action,
                                          deterministic_mass, random_mass)

    def public_action_likelihood(self, hand, view, public_event, choice):
        """Event and conditional physical probabilities from ONE detached plan.

        Keep the two factors separate so long-history callers can add their
        logs without underflowing their product first.
        """
        action, deterministic_mass, random_mass = self._conditional_components(hand, view, public_event)
        return (deterministic_mass+random_mass,
                self._physical_probability(hand, public_event, choice, action,
                                           deterministic_mass, random_mass))

    def public_action_likelihoods(self, hand, view, public_event, choices):
        """Score several physical packets using one detached plan.

        The hand, legal view and public event must be identical for all choices.
        Returns the same factors as repeated public_action_likelihood calls.
        """
        action, deterministic_mass, random_mass = self._conditional_components(hand, view, public_event)
        public = deterministic_mass + random_mass
        return [(public, self._physical_probability(hand, public_event, choice, action,
                    deterministic_mass, random_mass)) for choice in choices]

    @classmethod
    def _physical_probability(cls, hand, public_event, choice, action, deterministic_mass, random_mass):
        if not isinstance(choice, dict):
            raise ValueError('physical action must be a choice dict')
        if public_event[0] == 'play':
            cards = choice.get('cards')
            if (choice.get('action') != 'play' or choice.get('rank') != public_event[1]
                    or not isinstance(cards, (list, tuple)) or len(cards) != public_event[2]
                    or len(set(cards)) != len(cards) or any(c not in hand for c in cards)):
                return 0.
        elif choice.get('action') != ('call_bluff' if public_event[0] == 'challenge' else public_event[0]):
            return 0.
        if cls._event(choice, hand)[:3] != public_event:
            return 0.
        if public_event[0] != 'play':
            return 1.
        cards = choice['cards']
        exact = (deterministic_mass if tuple(action.get('cards', ())) == tuple(cards) else 0.)
        # Random.sample returns each ordered q-tuple with probability1/(n)_q.
        return (exact + random_mass/perm(len(hand), len(cards))) / (deterministic_mass+random_mass)

    def choose_action(self, hand, state):
        _validate_view(hand, state)
        self._set_viewer(state)
        self.observe_player_view(state)
        return self._choose_action_after_observe(hand,state)

    def _choose_action_after_observe(self,hand,state):
        # Preserve risk actor's hand/view validation even in trusted rollouts.
        _validate_view(hand,state)
        action, detail = self._plan(list(hand), state)
        if self.tremble and self.rng.random() < self.tremble:
            random_actor = RoundRandomBot(seed=0)
            random_actor.rng = self.rng
            random_actor.player_id = self.player_id
            action = random_actor.choose_action(list(hand), state)
            detail = {"reason": "risk_utility_tremble", "utility": None}
        self.last_decision_telemetry = {
            "policy": self.policy_name, "action": action["action"], **detail}
        return action

    def to_dict(self):
        parent = super().to_dict()
        parent['policy'] = RoundPickupAdaptiveBot.policy_name
        return {"version": self.version, "model_id": self.model_id,
                "ruleset_id": RULESET_ID,
                "config": {"risk_floor": self.risk_floor,
                           "tremble": self.tremble},
                "parent": parent}

    def restore_opponent_profile(self, data):
        if (not isinstance(data, dict)
                or set(data) != {"version", "model_id", "ruleset_id", "config", "parent"}
                or type(data.get("version")) is not int or data["version"] != self.version
                or data.get("model_id") != self.model_id
                or data.get("ruleset_id") != RULESET_ID
                or not isinstance(data.get("config"), dict)
                or set(data["config"]) != {"risk_floor", "tremble"}
                or any(type(v) not in (int, float) for v in data['config'].values())
                or data["config"] != {"risk_floor": self.risk_floor,
                                      "tremble": self.tremble}):
            raise ValueError("unsupported risk-utility actor profile")
        staged = RoundPickupAdaptiveBot(seed=0)
        if not isinstance(data['parent'], dict) or set(data['parent']) != set(staged.to_dict()):
            raise ValueError('unsupported parent actor profile')
        staged.restore_opponent_profile(deepcopy(data["parent"]))
        rng, player_id = self.rng, self.player_id
        self._call_evidence = staged._call_evidence
        self._bluff_evidence = staged._bluff_evidence
        self._resolved_action_objects.clear()
        self.last_decision_telemetry = {}
        self.rng, self.player_id = rng, player_id


__all__ = ["RoundRiskUtilityActor"]
