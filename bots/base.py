"""BotInterface — abstract base class for all Bluff bots.

Every bot (Random, Honest, CardCount, Adaptive, NN, Hybrid) implements this
interface. The game engine and web adapter only depend on this contract,
never on concrete bot implementations.

Design principle: the game engine calls decide_play() and decide_call().
The bot calls observe_action() to learn from every action it sees.
save/load persist learned state (opponent models, NN weights) across sessions.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from cards import Card, Rank
from game import Action
from bots.prob import Hypergeometric


class BotInterface(ABC):
    """Abstract base class for all Bluff bots."""

    @abstractmethod
    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        """Decide which cards to play and what rank to claim.

        Args:
            hand: Current cards in hand (sorted).
            game_state: Dict with:
                - opponent_hand_size (int)
                - pile_size (int)
                - draw_pile_size (int)
                - turn_number (int)
                - last_action (Action or None)
                - cards_played (List[Card]) — all revealed cards

        Returns:
            (cards_to_play, claimed_rank)
            Must play 1-4 cards. All cards must be from hand.
        """
        pass

    @abstractmethod
    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        """Decide whether to call bluff on the opponent's last play.

        Args:
            last_action: The action to evaluate (opponent's most recent play).
            game_state: Dict with:
                - hand_size (int) — our hand size
                - opponent_hand_size (int)
                - pile_size (int)
                - draw_pile_size (int)
                - turn_number (int)
                - cards_played (List[Card]) — all revealed cards

        Returns:
            True to call bluff, False to pass.
        """
        pass

    @abstractmethod
    def observe_action(self, action: Action, opponent_hand_size: int):
        """Observe any action (own or opponent's) for learning.

        Called after every play, call, or pass. Bots use this to update
        card counting, opponent models, etc.

        Args:
            action: The action that just occurred.
            opponent_hand_size: Opponent's hand size after this action.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Persist learned state to disk.

        Args:
            path: File path to save to (e.g., 'models/bayesian_user123.json').
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Load learned state from disk.

        Args:
            path: File path to load from.
        """
        pass

    def reset(self):
        """Reset bot state for a new game. Override if needed."""
        pass


def update_pending_claims(pending: dict, action: Action) -> None:
    """Track claimed cards sitting unresolved in the face-down pile.

    PUBLIC information only (no hidden-info leak): every uncalled play adds
    its CLAIMED rank/count to the pile; a call reveals and resolves the whole
    pile, clearing the pending tally.

    Used to build the pool model: pool[r] = 4 − own_hand[r] − pending[r].
    Unlike monotone "cards seen" counters, this self-heals when pile cards
    recycle into hands after a call — the old counters hit 0 for ranks still
    in play, which caused eternal call-war stalemates.
    """
    if action.bluff_called:
        for k in list(pending.keys()):
            pending[k] = 0
    else:
        pending[action.claimed_rank] = (
            pending.get(action.claimed_rank, 0) + len(action.cards_played)
        )


def pending_total(pending: dict) -> int:
    return sum(pending.values())


def pending_claims_from_actions(actions) -> dict:
    """Rebuild the pending-claims tally from the public action history.

    The claimed rank/count of every face-down play is public information;
    only the actual cards are hidden. A call resolves (reveals) the whole
    pile, so the tally resets there. Derived fresh each decision — this is
    the self-healing replacement for monotone 'cards seen' counters, which
    broke when pile cards recycled into hands after a call.
    """
    pending: dict = {}
    for action in actions:
        update_pending_claims(pending, action)
    return pending


def pool_by_rank(own_hand: List[Card], pending: dict) -> dict:
    """Copies of each rank available to {opponent hand + draw pile}.

    pool[r] = 4 − copies of r in our hand − unresolved pile claims of r.
    pool_total = 52 − our hand size − total pending claims.
    """
    own_counts = {rank: 0 for rank in Rank}
    for c in own_hand:
        if c.rank in own_counts:
            own_counts[c.rank] += 1
    return {rank: max(0, 4 - own_counts[rank] - pending.get(rank, 0))
            for rank in Rank}


def bluff_probability(own_hand: List[Card], pending: dict,
                      last_action: Action, opp_hand_size: int,
                      prior: float = 0.20, evidence_weight: float = 0.30) -> float:
    """P(opponent's last claim is a bluff) — posterior, not raw likelihood.

    v2 (2026-09-10, fixes the HonestBot call-everything degeneracy).

    v1 returned the raw random-null likelihood: P(a uniformly random hand of
    n pool cards would hold fewer than claim_size copies of the rank) — a
    hypergeometric CDF. Thresholding that as a posterior is a null-
    misspecification error: real opponents SELECT claims because they hold
    the cards (honest play) or mimic consistency (bluffing). Under that
    selection effect the random-hand null makes even fully honest multi-card
    claims look "improbable" (measured: honest 2-card claim → CDF 0.67,
    honest 3-card → 0.93), so any fixed threshold degenerated to
    call-everything late-game (HonestBot traced at 100% call rate incl.
    honest plays — base-rate neglect). See docs/decisions.md ADR.

    v2 semantics:
    - pool-inconsistent claim (fewer unseen copies than claimed):
      P = 1.0 — hard evidence, strategy-independent.
    - otherwise: P = (1−w)·prior + w·CDF, a shrunk blend of the population
      bluff base rate (0.2) with the old likelihood as a graded evidence
      term. Consistent claims stay below every consumer threshold
      (Honest 0.6, CardCount 0.4) while preserving size/late-game ordering.

    `prior`/`evidence_weight` are keyword-tunable for ablation (claim #2
    experiments) and are documented in docs/bot-modes.md §pool-model.
    """
    claim_size = len(last_action.cards_played)
    rank = last_action.claimed_rank

    pending_adj = dict(pending)
    pending_adj[rank] = max(0, pending_adj.get(rank, 0) - claim_size)

    pool = pool_by_rank(own_hand, pending_adj)
    pool_k = pool.get(rank, 0)
    pool_total = 52 - len(own_hand) - pending_total(pending_adj)
    n = opp_hand_size + claim_size  # their hand size before the play

    if pool_k < claim_size:
        return 1.0  # impossible claim — not enough unseen copies
    if pool_total <= 0:
        return 0.0
    n = min(n, pool_total)
    cdf = Hypergeometric.cdf(claim_size - 1, pool_total, pool_k, n)
    return (1.0 - evidence_weight) * prior + evidence_weight * cdf


def claim_plausibility(own_hand: List[Card], pending: dict, rank: Rank,
                       claim_size: int, opp_hand_size: int) -> float:
    """P(the opponent CANNOT disprove our claim of claim_size copies of
    `rank`) under the trust-model pool.

    Offensive math (pre-play: our claim is not yet in `pending`). The
    opponent can prove a bluff iff they hold more than
    4 − pending[r] − claim_size copies themselves, so
    P(survive) = Hypergeom CDF(4 − pending[r] − claim_size).
    """
    pool = pool_by_rank(own_hand, pending)
    pool_k = pool.get(rank, 0)
    pool_total = 52 - len(own_hand) - pending_total(pending)
    k = 4 - pending.get(rank, 0) - claim_size

    if k < 0:
        return 0.0  # claim exceeds all unseen copies — always disprovable
    if pool_total <= 0:
        return 1.0 if k >= pool_k else 0.0
    n = min(opp_hand_size, pool_total)
    return Hypergeometric.cdf(k, pool_total, pool_k, n)


def build_game_state(
    hand_size: int,
    opponent_hand_size: int,
    pile_size: int,
    draw_pile_size: int,
    turn_number: int,
    last_action: Optional[Action] = None,
    cards_played: Optional[List[Card]] = None,
    hand: Optional[List[Card]] = None,
    actions: Optional[List[Action]] = None,
) -> dict:
    """Helper to build the game_state dict passed to bots.

    Used by the game engine and web adapter to construct a consistent
    state dictionary from the current GameState.

    `hand` (the acting bot's own cards) is optional but recommended — the
    documented state contract (docs/bot-modes.md) includes it, and NN bots
    need it for state encoding when responding to a play.

    `actions` (the public action history) enables the pending-claims pool
    model: pass game.actions so probability bots can compute exact
    "copies left among {opponent hand + draw pile}" per rank. The history
    itself is also passed through (as `game_state["actions"]`) so bots with
    history-derived features (e.g. PureNNBot's v5 opponent signals) observe
    the same public information in tournaments as in training.
    """
    return {
        "hand_size": hand_size,
        "opponent_hand_size": opponent_hand_size,
        "pile_size": pile_size,
        "draw_pile_size": draw_pile_size,
        "turn_number": turn_number,
        "last_action": last_action,
        "cards_played": cards_played or [],
        "hand": hand or [],
        "pending_claims": pending_claims_from_actions(actions)
        if actions else {},
        "actions": list(actions) if actions else [],
    }
