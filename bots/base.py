"""BotInterface — abstract base class for all Bluff bots.

Every bot (Random, Honest, CardCount, Adaptive, NN, Hybrid) implements this
interface. The game engine and web adapter only depend on this contract,
never on concrete bot implementations.

Design principle: the game engine calls decide_play() and decide_call().
The bot calls observe_action() to learn from every action it sees.
save/load persist learned state (opponent models, NN weights) across sessions.
"""

import math
from copy import copy
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from cards import Card, Rank
from game import Action
from bots.prob import Hypergeometric

# Population bluff rate, used as the per-pile-claim poison probability when no
# opponent model exists (rule bots) and as the shrinkage prior for the learned
# one. Matches the ``prior`` of :func:`bluff_probability`; see ADR-040.
POPULATION_PILE_BLUFF_RATE = 0.20


def legalize_opening_claim(game, cards: List[Card], claimed_rank: Rank):
    """Project a bot's proposed opening onto the rules-legal action set.

    The locked rules require the first play to claim Aces. Later claims are
    unrestricted. This wrapper keeps legacy policies usable while ensuring
    the executed action is legal; neural policies should additionally mask
    their opening action space so training probabilities match execution.
    """
    if not game.actions:
        return cards, Rank.ACE
    return cards, claimed_rank


def opening_claim_pending(game_state: dict) -> bool:
    """Whether a state explicitly represents the game's first play."""
    if "opening_claim" in game_state:
        return bool(game_state["opening_claim"])
    return "actions" in game_state and not game_state.get("actions")


def legal_claim_ranks(game_state: dict):
    """Ranks a policy may claim in this state, honoring the opening rule."""
    return (Rank.ACE,) if opening_claim_pending(game_state) else tuple(Rank)


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

        Called after every play, call, or pass. An unchallenged opponent play
        is delivered as a public redacted action: card identities are None
        placeholders preserving packet size, and honesty is None. Bots use
        this to update card counting, opponent models, etc.

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


def guaranteed_honest_finish(
    hand: List[Card], *, first_claim: bool = False
) -> Optional[Tuple[List[Card], Rank]]:
    """Return the immediate winning play when the whole hand is one honest packet.

    A truthful play that empties the hand wins after either legal response:
    passing resolves the pending win, while calling gives the pile to the
    responder and still resolves it. Bluffing this exact hand away cannot help.
    """
    if (0 < len(hand) <= 4
            and all(card.rank == hand[0].rank for card in hand)
            and (not first_claim or hand[0].rank == Rank.ACE)):
        return list(hand), hand[0].rank
    return None


def public_bluff_label(action: Action, viewer: Optional[int]) -> Optional[bool]:
    """Honesty of ``action``, or ``None`` when ``viewer`` may not know it.

    game-rules.md §5 (information model) is explicit: a play's real cards are
    hidden **unless the play was challenged** (a call reveals them to both
    players), and they are of course always known to the player who played
    them. Every other read of ``Action.was_bluff`` is a hidden-information
    leak: it hands the agent the ground truth of the opponent's *uncalled*
    bluffs, which the game deliberately withholds.

    This is not a stylistic nicety. The flagship personalisation path
    (``bots.bayesian_bot.OpponentModel``) learned its opponent bluff rate from
    ``action.was_bluff`` on every observation. Measured consequence: vs
    HonestBot the model reached a 1:42 posterior after three games, i.e. it
    "knew" a never-bluffing opponent had a 2.3% bluff rate from evidence it
    could not have had. In human play no such label exists — a human's
    uncalled bluffs are never revealed — so the web bot was profiling a
    fantasy, and every "the model converges to 0% vs honest" result in the
    docs was measuring the leak rather than the model.

    Args:
        action: the action whose honesty is in question.
        viewer: the player asking (``None`` = an observer who owns no plays).

    Returns:
        ``True``/``False`` when the label is legally observable, else ``None``.
        Callers MUST treat ``None`` as "no evidence" rather than as honest.
    """
    if action.bluff_called:
        return bool(action.was_bluff)
    if viewer is not None and action.player == viewer:
        return bool(action.was_bluff)
    return None


def public_action_for_viewer(action: Optional[Action],
                             viewer: Optional[int]) -> Optional[Action]:
    """Detached action view with only information legally known to ``viewer``.

    An unchallenged opponent claim exposes its size/rank, not card identities
    or honesty. Keep ``len(cards_played)`` stable with ``None`` placeholders.
    """
    if action is None:
        return None
    public = copy(action)
    cards = list(getattr(action, "cards_played", ()) or ())
    visible = (not cards or action.bluff_called
               or (viewer is not None and action.player == viewer))
    public.cards_played = cards if visible else [None] * len(cards)
    if not visible:
        public.was_bluff = None
        public.caller_was_right = False
    response_to = getattr(action, "response_to", None)
    if response_to is not None:
        public.response_to = public_action_for_viewer(response_to, viewer)
    return public


def public_actions_for_viewer(actions, viewer: Optional[int]) -> list[Action]:
    """Detached, information-set-safe view of an action history."""
    return [public_action_for_viewer(action, viewer) for action in actions]


class HandCacheMixin:
    """Remembers the agent's own most recent hand across decide/observe calls.

    ``BotInterface.observe_action`` has no hand argument, but the opponent
    model's conflict context needs *our* copies of a claimed rank. Conflict is
    a useful predictor for calls and bluff labels, but it is not necessarily a
    sufficient description of the call policy or a correction for selective
    labels. Our own cards are legal knowledge, so caching them leaks nothing.

    The cache is refreshed on every decision (``decide_play`` /
    ``decide_call``), and a decision is always the immediately preceding event
    for the claim being observed, so the cached hand is the one the claim was
    actually judged against.

    It also caches the draw-pile size, which is how a bot can tell a
    *voluntary* challenge from a mandatory one: passing is illegal once the
    draw pile is empty (game-rules.md §2), so a challenge there is not a
    choice and says nothing about the opponent's willingness to call. That is
    the same voluntary/forced split ``analysis/bluff_eval.py`` makes, and
    feeding forced calls into a challenge-rate estimate drives it to 100% in
    the endgame.

    Bots using this must still set ``self._hand_cache = []`` and
    ``self._draw_pile_cache = 1`` in ``__init__``.
    """

    _hand_cache: List[Card]
    _draw_pile_cache: int

    def _cache_hand(self, game_state: dict,
                    hand: Optional[List[Card]] = None) -> List[Card]:
        """Refresh and return the cached own hand from a decision state.

        Args:
            game_state: the decision state.
            hand: the acting bot's own cards, when the caller has them as an
                argument (``decide_play``); otherwise taken from the state.
        """
        if hand is None:
            hand = game_state.get("hand") or []
        if hand:
            self._hand_cache = list(hand)
        if "draw_pile_size" in game_state:
            self._draw_pile_cache = int(game_state.get("draw_pile_size") or 0)
        return getattr(self, "_hand_cache", []) or []

    def _response_was_forced(self) -> bool:
        """True when the last response we know of was mandatory.

        A challenge is forced once the draw pile is empty, so it carries no
        information about the opponent's calling *preference*.
        """
        return int(getattr(self, "_draw_pile_cache", 1)) <= 0

    def _my_copies(self, action: Action) -> Optional[int]:
        """Our copies of ``action``'s claimed rank, or ``None`` if unknown."""
        if not action.cards_played:
            return None
        cache = getattr(self, "_hand_cache", None)
        if not cache:
            return None
        return sum(1 for c in cache if c.rank == action.claimed_rank)


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


def refresh_pool(counter, own_hand: List[Card], game_state: dict,
                 last_action: Optional[Action] = None) -> None:
    """Self-heal a ``CardCounter`` from PUBLIC information (ADR-026).

    Call once per decision, before any counter query. Reproducing this file's
    own pool model is the point: ``bots.base.bluff_probability`` was written to
    be self-healing precisely because a monotone "cards seen" tally breaks when
    a call recycles the pile into a hand, and the counter class in
    ``bots/bayesian_bot`` still had that defect — it double-subtracted absorbed
    cards until every rank's pool hit 0, at which point every multi-card claim
    read as a *certain* bluff.

    ``last_action`` identifies the claim currently under evaluation so its
    cards are not counted as living in the opponent's hand.
    """
    pending = game_state.get("pending_claims")
    if pending is None:
        pending = pending_claims_from_actions(game_state.get("actions") or [])
    rank = last_action.claimed_rank if last_action is not None else None
    size = len(last_action.cards_played) if last_action is not None else 0
    counter.rebuild(own_hand, pending, rank, size)


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


def graded_bluff_posterior(pool_k: int, claim_size: int, pool_total: int,
                           n: int, prior: float = 0.20,
                           evidence_weight: float = 0.30) -> float:
    """P(bluff) from the pool alone, when ``pool_k`` may be known exactly.

    This is the graded branch of :func:`bluff_probability`: ``pool_k`` copies
    of the claimed rank are *certainly* unavailable to the opponent, so a
    claim of more than that many is impossible (P = 1.0), and otherwise the
    random-hand likelihood is shrunk toward the population base rate.
    """
    if pool_total <= 0:
        return 1.0 if pool_k < claim_size else 0.0
    if pool_k < claim_size:
        return 1.0  # impossible claim — not enough unseen copies
    n = min(n, pool_total)
    cdf = Hypergeometric.cdf(claim_size - 1, pool_total, pool_k, n)
    return (1.0 - evidence_weight) * prior + evidence_weight * cdf


def pile_uncertain_bluff_posterior(pool_k: int, claim_size: int,
                                   pool_total: int, n: int,
                                   pending_r: int, pile_bluff_rate: float,
                                   prior: float = 0.20,
                                   evidence_weight: float = 0.30) -> float:
    """P(bluff) when the pool test may be poisoned by an uncalled pile lie.

    ``pool[r] = 4 - our copies of r - unresolved pile claims of r`` treats an
    unresolved pile claim of r as a *whole* card of r. But a pile claim is
    only a claim (ADR-039): if the opponent bluffed it, the real card is not
    r at all, the pool is understated, and ``pool_k < claim_size`` stops being
    proof. Measured, that error is exact: certain calls against a clean pile
    were **0.0% wrong** (n=215), against a same-rank-poisoned pile **22.4%**
    (n=76). The exploit is *lie once into rank r (uncalled), then play r
    honestly* and collect the pile the counter thought it could prove.

    Fix: treat each unresolved pile claim of the rank as worth ``1 - b`` of a
    card instead of a whole one, where ``b`` is the opponent's bluff rate, and
    marginalise. ``D ~ Binomial(pending_r, b)`` is how many of those claimed
    cards were lies; the claim is possible iff ``D >= claim_size - pool_k``.
    Each branch is scored by the graded pool model with ``pool_k + D`` copies
    available, so certainty degrades to evidence and nothing else changes.

    ``pile_bluff_rate=0.0`` reproduces the legacy hard rule exactly (only
    ``D = 0`` has mass), which is what keeps it backward compatible.
    """
    if pool_k >= claim_size or pending_r <= 0 or pile_bluff_rate <= 0.0:
        return graded_bluff_posterior(pool_k, claim_size, pool_total, n,
                                      prior, evidence_weight)
    b = min(1.0, max(0.0, float(pile_bluff_rate)))
    total = 0.0
    for d in range(0, pending_r + 1):
        pmf = (math.comb(pending_r, d) * (b ** d)
               * ((1.0 - b) ** (pending_r - d)))
        if pmf <= 0.0:
            continue
        total += pmf * graded_bluff_posterior(
            pool_k + d, claim_size, pool_total, n, prior, evidence_weight)
    return total


def bluff_probability(own_hand: List[Card], pending: dict,
                      last_action: Action, opp_hand_size: int,
                      prior: float = 0.20, evidence_weight: float = 0.30,
                      pile_bluff_rate: float = 0.0) -> float:
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

    ``pile_bluff_rate`` (ADR-040) is the probability that one unresolved pile
    claim of the *same rank* was a lie, which is what makes the ``pool_k <
    claim_size`` branch sound only when it is 0.0 (the default, so every
    existing caller keeps the legacy semantics; the training encoders in
    ``nn/state_encoder.py`` and ``nn/grid_encoder.py`` must stay at 0.0 or
    their features and the shipped checkpoints drift apart). A deployed bot
    passes the opponent's measured bluff rate so one pile lie can no longer
    turn a later honest play of that rank into a free pile.
    """
    claim_size = len(last_action.cards_played)
    rank = last_action.claimed_rank

    pending_adj = dict(pending)
    pending_adj[rank] = max(0, pending_adj.get(rank, 0) - claim_size)

    pool = pool_by_rank(own_hand, pending_adj)
    pool_k = pool.get(rank, 0)
    pool_total = 52 - len(own_hand) - pending_total(pending_adj)
    n = opp_hand_size + claim_size  # their hand size before the play

    return pile_uncertain_bluff_posterior(
        pool_k, claim_size, pool_total, n, pending_adj.get(rank, 0),
        pile_bluff_rate, prior, evidence_weight)


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
    viewer: Optional[int] = None,
) -> dict:
    """Helper to build the game_state dict passed to bots.

    Used by the game engine and web adapter to construct a consistent
    state dictionary from the current GameState.

    `hand` (the acting bot's own cards) is optional but recommended — the
    documented state contract (docs/bot-modes.md) includes it, and NN bots
    need it for state encoding when responding to a play.

    `actions` (the engine's action history) enables the pending-claims pool
    model: pass game.actions so probability bots can compute exact
    "copies left among {opponent hand + draw pile}" per rank. The history
    is copied into a legal observation for ``viewer`` before any bot sees it.

    An omitted viewer gets public-only information; own unchallenged cards
    are retained only when their player id is supplied. Never pass
    ``GameState.get_cards_played()`` through here — it includes hidden cards.
    """
    public_actions = public_actions_for_viewer(actions or [], viewer)
    public_last_action = public_action_for_viewer(last_action, viewer)
    if actions:
        cards_played = [c for a in public_actions
                        if a.cards_played and a.bluff_called
                        for c in a.cards_played]
    pending = pending_claims_from_actions(public_actions) if public_actions else {}
    return {
        "hand_size": hand_size,
        "opponent_hand_size": opponent_hand_size,
        "pile_size": pile_size,
        "draw_pile_size": draw_pile_size,
        "turn_number": turn_number,
        "opening_claim": not bool(public_actions),
        "last_action": public_last_action,
        "cards_played": cards_played or [],
        "hand": hand or [],
        "pending_claims": pending,
        "actions": public_actions,
        # Copies of each rank still available to {opponent hand + draw pile},
        # i.e. the pool model of docs/bot-modes.md. Legal (own hand + public
        # pending claims) and, until now, only ever populated by NNPlayer's
        # private context builder — so every offline encoder that consumed
        # build_game_state (the dataset collector included) silently saw the
        # neutral default 4 for all 13 ranks, i.e. 13 dead features of 39.
        "cards_remaining": pool_by_rank(hand or [], pending),
    }
