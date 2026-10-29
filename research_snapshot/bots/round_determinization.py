"""Research-only root proposals, not a history-conditioned belief posterior.

Uniformly complete current card zones subject to own visible cards, public
sizes and conservative opponent rank bounds. Optional latest-packet truth
strata support SDMCTS-style comparisons. These samples deliberately do NOT
model opponent action selection or all historical ownership correlations;
their frequencies must not be advertised as calibrated bluff probabilities.
No live GameState or privileged opponent state is accepted. Optional required
identities are caller-supplied legal evidence for round-start replay proposals,
not permission to inspect an opponent hand. allocation_count counts physical
set allocations across hand/latest/older-hidden-pile/draw zones; zone permutations are sampled conditionally uniformly.
"""

from dataclasses import dataclass
from collections import Counter
from functools import lru_cache
from math import comb
from random import Random

from bots.round_card_memory import opponent_rank_bounds, opponent_hand_and_center_bounds
from cards import Card, Rank, Suit


@dataclass(frozen=True)
class RootWorld:
    opponent_hand: tuple[Card, ...]
    pile: tuple[Card, ...]
    draw_supply: tuple[Card, ...]
    pending_packet: tuple[Card, ...]
    allocation_count: int  # uniform proposal cardinality;0=undefined for direct posterior draws


def sample_root_world(hand, state, rng: Random, *, pending_truth=None,
                      max_states=10000, opponent_pile_truthful=False,
                      preclaim_rank_caps=None,fixed_active_counts=None,fixed_hand_rank=None,
                      fixed_hand_groups=None, required_opponent_cards=()):
    """Return a proposal, or fail explicitly if constraints/budget cannot hold.

    Count weighted rank allocations, then sample physical cards uniformly.
    This avoids rejection failures when a small hand has tight known bounds.
    Exhausting the finite DP budget is not evidence a stratum is impossible.
    required_opponent_cards conditions a ROUND-START hypothetical hand only.
    It does not update behavioral likelihoods or repair a posterior by itself.
    """
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("invalid viewer")
    if pending_truth is not None and type(pending_truth) is not bool:
        raise ValueError("pending_truth must be bool or None")
    if fixed_active_counts is not None and (
            not isinstance(fixed_active_counts,tuple) or len(fixed_active_counts)!=2
            or any(type(n) is not int or not 0<=n<=4 for n in fixed_active_counts)
            or sum(fixed_active_counts)>4):
        raise ValueError('fixed_active_counts must be physical (hand,latest) counts')
    if fixed_hand_rank is not None and (
            not isinstance(fixed_hand_rank,tuple) or len(fixed_hand_rank)!=2
            or type(fixed_hand_rank[0]) is not Rank or type(fixed_hand_rank[1]) is not int
            or not 0<=fixed_hand_rank[1]<=4):
        raise ValueError('fixed_hand_rank must be (canonical rank, physical count)')
    if type(opponent_pile_truthful) is not bool:
        raise ValueError("opponent_pile_truthful must be bool")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be positive")
    if preclaim_rank_caps is not None and (not isinstance(preclaim_rank_caps, dict) or
        any(type(r) is not Rank or type(n) is not int or not 0 <= n <= 4
            for r, n in preclaim_rank_caps.items())):
        raise ValueError("preclaim rank caps must map canonical ranks to0..4")
    hand = tuple(hand)
    sizes = [state.get(key) for key in
             ("hand_size", "opponent_hand_size", "pile_size", "draw_pile_size")]
    if any(type(n) is not int or n < 0 for n in sizes):
        raise ValueError("invalid public card counts")
    own_size, opponent_size, pile_size, draw_size = sizes
    if fixed_hand_groups is not None and (type(fixed_hand_groups) is not int
            or not 0<=fixed_hand_groups<=min(13,opponent_size)):
        raise ValueError('fixed_hand_groups must be a feasible distinct-rank count')
    if own_size != len(hand) or sum(sizes) != 52:
        raise ValueError("public counts must conserve 52 cards")
    if fixed_hand_rank is not None and fixed_hand_rank[1]>opponent_size:
        raise ValueError('fixed hand-rank count exceeds opponent hand size')
    current = [action for action in state.get("actions", ())
               if action.round_number == state.get("round_number")]
    slots = []
    for action in current:
        cards = tuple(action.cards_played)
        slots.extend(cards if action.player == viewer else (None,) * len(cards))
    if len(slots) != pile_size:
        raise ValueError("current-round packets must match pile size")
    known_pile = tuple(card for card in slots if card is not None)
    if known_pile != tuple(state.get("own_pile_cards", ())):
        raise ValueError("own pile cards disagree with current history")
    deck = tuple(Card(rank, suit) for suit in Suit for rank in Rank)
    known = hand + known_pile
    if any(card not in deck for card in known) or len(set(known)) != len(known):
        raise ValueError("known current card locations must be valid and distinct")
    if (not isinstance(required_opponent_cards,(list,tuple))
            or any(type(card) is not Card or type(card.rank) is not Rank
                   or type(card.suit) is not Suit or card not in deck for card in required_opponent_cards)
            or len(set(required_opponent_cards))!=len(required_opponent_cards)
            or len(required_opponent_cards)>opponent_size
            or set(required_opponent_cards).intersection(known)):
        raise ValueError('required opponent cards must be distinct physical cards outside own known zones')
    required_opponent_cards=tuple(required_opponent_cards)
    if required_opponent_cards and (pile_size or state.get('active_rank') is not None
                                    or state.get('last_action') is not None):
        raise ValueError('required opponent identities currently support round-start proposals only')
    required_counts=Counter(card.rank for card in required_opponent_cards)
    pending = state.get("last_action")
    quantity = 0
    if pending is not None:
        if (not current or pending.action_index != current[-1].action_index
                or pending.player == viewer):
            raise ValueError("pending packet must be the latest opponent play")
        quantity = len(pending.cards_played)
        if not 1 <= quantity <= 4 or not isinstance(pending.claimed_rank, Rank):
            raise ValueError("invalid pending claim")
    if pending_truth is not None and pending is None:
        raise ValueError("truth stratification requires a pending packet")
    if fixed_active_counts is not None and (pending is None
            or fixed_active_counts[0]>opponent_size or fixed_active_counts[1]>quantity):
        raise ValueError('active-count conditioning requires compatible pending zones')
    pool = [card for card in deck if card not in known and card not in required_opponent_cards]
    # A truthful-policy hypothesis conditions every earlier hidden packet in
    # this fixed-rank round. Reserve those cards before sampling current hand
    # and pending packet so zones stay disjoint and card constraints remain exact.
    reserved = ()
    reserve_ways = 1
    if opponent_pile_truthful:
        if pending is not None:
            if pending_truth is False:
                raise ValueError("truthful-policy hypothesis contradicts false pending stratum")
            pending_truth = True
        older = slots[:-quantity] if quantity else slots
        needed = sum(card is None for card in older)
        available = [card for card in pool if card.rank == state.get("active_rank")]
        if needed > len(available):
            raise ValueError("truthful earlier-pile hypothesis has no support")
        reserve_ways = comb(len(available), needed)
        reserved = tuple(rng.sample(available, needed))
        pool = [card for card in pool if card not in reserved]
    rank_pool = [card for card in pool
                 if pending is not None and card.rank == pending.claimed_rank]
    if pending_truth is True and len(rank_pool) < quantity:
        raise ValueError("truthful stratum contradicts known current cards")
    bounds, before_bounds = opponent_rank_bounds(state.get("actions"), viewer, pending)
    combined_bounds = opponent_hand_and_center_bounds(
        state.get("actions"), viewer, state.get("round_number"))
    reserved_counts = Counter(card.rank for card in reserved)
    older_slots = slots[:-quantity] if quantity else slots
    older_size = sum(card is None for card in older_slots) if not opponent_pile_truthful else 0
    ranks = tuple(Rank)
    rank_cards = [tuple(card for card in pool if card.rank == rank) for rank in ranks]
    visited = 0

    def choices(index, own_left, packet_left, older_left, mismatch):
        rank, available = ranks[index], len(rank_cards[index])
        required=required_counts[rank]
        for own in range(max(0,bounds.get(rank, 0)-required), min(available, own_left) + 1):
            if fixed_hand_rank is not None and rank==fixed_hand_rank[0] and own+required!=fixed_hand_rank[1]:
                continue
            for packet in range(min(available - own, packet_left) + 1):
                if (fixed_active_counts is not None and pending is not None
                        and rank==pending.claimed_rank and (own+required,packet)!=fixed_active_counts):
                    continue
                if preclaim_rank_caps is not None and own+required+packet > preclaim_rank_caps.get(rank, 4):
                    continue
                if own + required + packet < before_bounds.get(rank, 0):
                    continue
                wrong = bool(packet and pending is not None and rank != pending.claimed_rank)
                if pending_truth is True and wrong:
                    continue
                for older in range(min(available-own-packet, older_left)+1):
                    if own+required+packet+older+reserved_counts[rank] < combined_bounds.get(rank,0):
                        continue
                    yield (own, packet, older, mismatch or wrong,
                           comb(available, own)*comb(available-own, packet)
                           *comb(available-own-packet, older))

    @lru_cache(maxsize=None)
    def completions(index, own_left, packet_left, older_left, mismatch, groups_left):
        nonlocal visited
        visited += 1
        if visited > max_states:
            raise RuntimeError("root proposal state budget exhausted; constraints were not relaxed")
        if groups_left < -1 or (fixed_hand_groups is not None
                and (groups_left<0 or groups_left>own_left+sum(required_counts[r]>0 for r in ranks[index:])
                     or groups_left>len(ranks)-index)):
            return 0
        if index == len(ranks):
            return int(own_left == packet_left == older_left == 0
                       and (fixed_hand_groups is None or groups_left==0)
                       and (pending_truth is not False or mismatch))
        return sum(weight * completions(index + 1, own_left - own,
                                        packet_left - packet, older_left - older, changed,
                                        groups_left-int(own+required_counts[ranks[index]]>0) if fixed_hand_groups is not None else -1)
                   for own, packet, older, changed, weight
                   in choices(index, own_left, packet_left, older_left, mismatch))

    groups_left = -1 if fixed_hand_groups is None else fixed_hand_groups
    residual_opponent_size=opponent_size-len(required_opponent_cards)
    total = completions(0, residual_opponent_size, quantity, older_size, False, groups_left)
    if not total:
        raise ValueError("no root proposal satisfies current card constraints")
    allocation_count = total * reserve_ways
    opponent, packet_cards, older_cards, residual = list(required_opponent_cards), [], [], []
    own_left, packet_left, older_left, mismatch = residual_opponent_size, quantity, older_size, False
    for index, cards in enumerate(rank_cards):
        ticket = rng.randrange(total)
        for own, packet, older, changed, weight in choices(index, own_left, packet_left, older_left, mismatch):
            next_groups = groups_left-int(own+required_counts[ranks[index]]>0) if fixed_hand_groups is not None else -1
            tail = completions(index + 1, own_left - own, packet_left - packet, older_left - older, changed,next_groups)
            branch = weight * tail
            if ticket >= branch:
                ticket -= branch
                continue
            chosen = rng.sample(cards, own)
            others = [card for card in cards if card not in chosen]
            played = rng.sample(others, packet)
            opponent.extend(chosen)
            packet_cards.extend(played)
            unplayed = [card for card in others if card not in played]
            earlier = rng.sample(unplayed, older)
            older_cards.extend(earlier)
            residual.extend(card for card in unplayed if card not in earlier)
            own_left, packet_left, older_left, mismatch = own_left-own, packet_left-packet, older_left-older, changed
            groups_left = next_groups
            total = tail
            break
    rng.shuffle(opponent)
    rng.shuffle(packet_cards)
    rng.shuffle(older_cards)
    rng.shuffle(residual)
    hidden = iter(residual)
    older_slots = slots[:-quantity] if quantity else slots
    reserved_iter = iter(reserved if opponent_pile_truthful else older_cards)
    pile = tuple(next(reserved_iter)
                 if card is None else card for card in older_slots)
    pile += tuple(packet_cards)
    return RootWorld(tuple(opponent), pile, tuple(hidden), tuple(packet_cards), allocation_count)
