"""BayesianBot — Bayesian opponent model + card counting + offensive bluffing.

The main competitive bot. Combines:
1. Card counting (hypergeometric P(bluff))
2. Bayesian opponent modeling (Beta distributions per feature)
3. Adaptive decision fusion (trusts model more as data accumulates)
4. Offensive bluffing: self-bluff tracking + adaptive bluff rate

Persists learned state per opponent for cross-game learning.
"""

import json
import os
import random
from typing import List, Tuple, Dict, Optional
from cards import Card, Rank
from game import Action
from bots.base import BotInterface, bluff_probability as pool_bluff_prob
from bots.prob import Hypergeometric


class BetaDistribution:
    """Simple Beta distribution for Bayesian updating."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Variance of Beta(alpha, beta) = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.0
        return (self.alpha * self.beta) / ((total ** 2) * (total + 1.0))

    def sample(self) -> float:
        """Draw a sample from the Beta posterior (Thompson Sampling)."""
        return random.betavariate(max(1e-3, self.alpha), max(1e-3, self.beta))

    def update(self, was_bluff: bool):
        if was_bluff:
            self.alpha += 1
        else:
            self.beta += 1

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> "BetaDistribution":
        return cls(alpha=d["alpha"], beta=d["beta"])


class OpponentModel:
    """Bayesian opponent model using Beta distributions with hierarchical shrinkage."""

    def __init__(self):
        self.bluff_by_hand_size: Dict[int, BetaDistribution] = {
            i: BetaDistribution() for i in range(1, 27)
        }
        self.bluff_by_rank: Dict[Rank, BetaDistribution] = {
            rank: BetaDistribution() for rank in Rank
        }
        self.bluff_by_claim_size: Dict[int, BetaDistribution] = {
            i: BetaDistribution() for i in range(1, 5)
        }
        self.call_frequency = BetaDistribution()
        self.overall_bluff = BetaDistribution(alpha=1.0, beta=4.0)  # prior base rate ~20%
        self.total_actions_observed = 0

    def observe_action(self, action: Action, opponent_hand_size: int):
        self.total_actions_observed += 1

        # Passes arrive as fabricated Actions with empty cards_played
        # (server.py handle_human_pass). A pass carries NO evidence about
        # rank honesty — recording it as "rank TWO, not bluff" polluted the
        # rank/claim-size models with fake honest observations. Only the
        # call_frequency signal (chose NOT to call) is real evidence.
        if not action.cards_played:
            self.call_frequency.update(action.bluff_called)
            return

        self.overall_bluff.update(action.was_bluff)

        hs = min(opponent_hand_size, 26)
        if hs not in self.bluff_by_hand_size:
            self.bluff_by_hand_size[hs] = BetaDistribution()
        self.bluff_by_hand_size[hs].update(action.was_bluff)

        self.bluff_by_rank[action.claimed_rank].update(action.was_bluff)

        cs = min(len(action.cards_played), 4)
        if cs not in self.bluff_by_claim_size:
            self.bluff_by_claim_size[cs] = BetaDistribution()
        self.bluff_by_claim_size[cs].update(action.was_bluff)

        if action.bluff_called:
            self.call_frequency.update(True)
        else:
            self.call_frequency.update(False)

    def estimate_bluff_probability(self, hand_size: int, claimed_rank: Rank,
                                   claim_size: int) -> float:
        estimates = []
        weights = []

        hs = min(hand_size, 26)
        if hs in self.bluff_by_hand_size:
            dist = self.bluff_by_hand_size[hs]
            if (dist.alpha + dist.beta) > 2.0:
                estimates.append(dist.mean())
                weights.append(2.0)

        if claimed_rank in self.bluff_by_rank:
            dist = self.bluff_by_rank[claimed_rank]
            if (dist.alpha + dist.beta) > 2.0:
                estimates.append(dist.mean())
                weights.append(1.5)

        cs = min(claim_size, 4)
        if cs in self.bluff_by_claim_size:
            dist = self.bluff_by_claim_size[cs]
            if (dist.alpha + dist.beta) > 2.0:
                estimates.append(dist.mean())
                weights.append(1.0)

        global_p = self.overall_bluff.mean()
        if not estimates:
            return global_p

        local_p = sum(e * w for e, w in zip(estimates, weights)) / sum(weights)
        total_w = sum(weights)
        shrinkage = min(0.70, total_w / 5.0)
        return (1.0 - shrinkage) * global_p + shrinkage * local_p

    def estimate_call_frequency(self) -> float:
        return self.call_frequency.mean()

    def sample_bluff_probability(self, hand_size: int, claimed_rank: Rank,
                                  claim_size: int) -> float:
        """Sample bluff probability via Thompson Sampling from posterior Beta distributions."""
        samples = []
        weights = []

        hs = min(hand_size, 26)
        if hs in self.bluff_by_hand_size:
            dist = self.bluff_by_hand_size[hs]
            if (dist.alpha + dist.beta) > 2.0:
                samples.append(dist.sample())
                weights.append(2.0)

        if claimed_rank in self.bluff_by_rank:
            dist = self.bluff_by_rank[claimed_rank]
            if (dist.alpha + dist.beta) > 2.0:
                samples.append(dist.sample())
                weights.append(1.5)

        cs = min(claim_size, 4)
        if cs in self.bluff_by_claim_size:
            dist = self.bluff_by_claim_size[cs]
            if (dist.alpha + dist.beta) > 2.0:
                samples.append(dist.sample())
                weights.append(1.0)

        global_p = self.overall_bluff.sample()
        if not samples:
            return global_p

        local_p = sum(s * w for s, w in zip(samples, weights)) / sum(weights)
        total_w = sum(weights)
        shrinkage = min(0.70, total_w / 5.0)
        return (1.0 - shrinkage) * global_p + shrinkage * local_p

    def sample_call_frequency(self) -> float:
        """Sample call frequency via Thompson Sampling."""
        return self.call_frequency.sample()

    def to_dict(self) -> dict:
        return {
            "total_actions_observed": self.total_actions_observed,
            "call_frequency": self.call_frequency.to_dict(),
            "overall_bluff": self.overall_bluff.to_dict(),
            "bluff_by_hand_size": {str(k): v.to_dict()
                                   for k, v in self.bluff_by_hand_size.items()},
            "bluff_by_rank": {str(k.value): v.to_dict()
                              for k, v in self.bluff_by_rank.items()},
            "bluff_by_claim_size": {str(k): v.to_dict()
                                    for k, v in self.bluff_by_claim_size.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpponentModel":
        model = cls()
        model.total_actions_observed = d.get("total_actions_observed", 0)
        model.call_frequency = BetaDistribution.from_dict(d["call_frequency"])
        if "overall_bluff" in d:
            model.overall_bluff = BetaDistribution.from_dict(d["overall_bluff"])
        for k, v in d.get("bluff_by_hand_size", {}).items():
            model.bluff_by_hand_size[int(k)] = BetaDistribution.from_dict(v)
        for k, v in d.get("bluff_by_rank", {}).items():
            model.bluff_by_rank[Rank(int(k))] = BetaDistribution.from_dict(v)
        for k, v in d.get("bluff_by_claim_size", {}).items():
            model.bluff_by_claim_size[int(k)] = BetaDistribution.from_dict(v)
        return model

    def apply_session_decay(self, lam: float) -> None:
        """Prior dilution for cross-session persistence (T8 profiling).

        Shrinks every Beta toward its fresh prior: B' = lam*B + (1-lam)*B0,
        bounding poisoning/staleness impact. lam in [0.5, 1.0] (1.0 = none).
        Fresh priors mirror __init__: overall_bluff (1,4), rest (1,1).
        total_actions_observed scales proportionally (TD-MoE re-ramps).
        """
        def _dilute(b: BetaDistribution, a0: float, b0: float) -> None:
            b.alpha = lam * b.alpha + (1.0 - lam) * a0
            b.beta = lam * b.beta + (1.0 - lam) * b0

        _dilute(self.overall_bluff, 1.0, 4.0)
        _dilute(self.call_frequency, 1.0, 1.0)
        for dist in (self.bluff_by_hand_size, self.bluff_by_rank,
                     self.bluff_by_claim_size):
            for bb in dist.values():
                _dilute(bb, 1.0, 1.0)
        self.total_actions_observed = int(self.total_actions_observed * lam)


class BluffTracker:
    """Tracks the bot's own bluff outcomes for adaptive bluff rate.

    Maintains a sliding window of recent bluffs to estimate:
    - Current bluff success rate
    - Whether opponent is catching on
    - Adaptive bluff frequency based on Nash equilibrium
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.outcomes: List[bool] = []  # True = got away with it
        self.total_attempted = 0
        self.total_caught = 0
        self.total_got_through = 0

    def record_bluff(self, was_called: bool):
        """Record the outcome of a bluff attempt."""
        got_through = not was_called
        self.outcomes.append(got_through)
        if len(self.outcomes) > self.window_size:
            self.outcomes.pop(0)
        self.total_attempted += 1
        if was_called:
            self.total_caught += 1
        else:
            self.total_got_through += 1

    def success_rate(self) -> float:
        """Bluff success rate over the sliding window."""
        if not self.outcomes:
            return 0.5  # no data, assume 50/50
        return sum(self.outcomes) / len(self.outcomes)

    def nash_bluff_rate(self, pile_size: int) -> float:
        """Nash equilibrium bluff frequency given current pile size.

        When pile is large, caller benefits more from calling → bluff less.
        When pile is small, bluffer benefits more → bluff more.
        """
        return 1.0 / (pile_size + 1)

    def adaptive_bluff_rate(self, pile_size: int,
                            opponent_call_freq: float) -> float:
        """Compute target bluff rate combining Nash + opponent model.

        If opponent calls a lot → bluff less (they're aggressive).
        If opponent rarely calls → bluff more (they're passive).
        Nash provides the baseline; opponent frequency adjusts it.
        """
        nash = self.nash_bluff_rate(pile_size)
        # Strong adjustment: aggressive callers need much lower bluff rate
        if opponent_call_freq > 0.6:
            adjustment = 0.3  # aggressive caller → slash bluff rate
        elif opponent_call_freq > 0.4:
            adjustment = 0.6  # moderate caller → reduce
        elif opponent_call_freq < 0.2:
            adjustment = 1.5  # passive caller → bluff more
        else:
            adjustment = 1.0  # neutral
        return max(0.02, min(0.4, nash * adjustment))

    def should_bluff(self, hand_size: int, opponent_hand_size: int,
                     pile_size: int, opponent_call_freq: float) -> bool:
        """Decide whether to bluff this turn based on adaptive rate."""
        target_rate = self.adaptive_bluff_rate(
            pile_size, opponent_call_freq
        )

        # Urgency: bluff more when close to winning (few cards left)
        urgency = 1.0
        if hand_size <= 3:
            urgency = 1.5
        elif hand_size <= 5:
            urgency = 1.2

        # Desperation: bluff more when opponent is close to winning
        if opponent_hand_size <= 3:
            urgency *= 1.3

        # Current success rate affects willingness
        sr = self.success_rate()
        if sr < 0.2:
            urgency *= 0.5  # getting caught a lot → bluff less
        elif sr > 0.7:
            urgency *= 1.3  # getting away with it → can bluff more

        effective_rate = min(0.7, target_rate * urgency)
        return random.random() < effective_rate

    def to_dict(self) -> dict:
        return {
            "outcomes": self.outcomes,
            "total_attempted": self.total_attempted,
            "total_caught": self.total_caught,
            "total_got_through": self.total_got_through,
            "window_size": self.window_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BluffTracker":
        bt = cls(window_size=d.get("window_size", 50))
        bt.outcomes = d.get("outcomes", [])
        bt.total_attempted = d.get("total_attempted", 0)
        bt.total_caught = d.get("total_caught", 0)
        bt.total_got_through = d.get("total_got_through", 0)
        return bt


class CardCounter:
    """Tracks visible cards for exact probability calculations."""

    def __init__(self):
        self.remaining_by_rank: Dict[Rank, int] = {rank: 4 for rank in Rank}
        self.total_remaining: int = 52

    def update_with_play(self, cards: List[Card]):
        for card in cards:
            if self.remaining_by_rank[card.rank] > 0:
                self.remaining_by_rank[card.rank] -= 1
                self.total_remaining -= 1

    def bluff_probability(self, claimed_rank: Rank, claim_size: int,
                          our_hand_size: int, opp_hand_size: int,
                          our_copies: int = 0,
                          prior: float = 0.20,
                          evidence_weight: float = 0.30) -> float:
        """P(opponent is lying) — posterior, not raw likelihood.

        v2 (2026-09-10, ported from bots/base.py bluff_probability shrinkage).

        Identical semantics to the pool-model version in base.py:
        - pool-inconsistent claim (fewer unseen copies than claimed):
          P = 1.0 — hard evidence, strategy-independent.
        - otherwise: P = (1−w)·prior + w·CDF, a shrunk blend of the
          population bluff base rate (0.20) with the hypergeometric
          likelihood as a graded evidence term.

        Args:
            our_copies: copies of claimed_rank in OUR hand — subtracted from
                the unseen pool since the opponent cannot hold them.
            prior: population bluff base rate (default 0.20).
            evidence_weight: shrinkage weight on the CDF (default 0.30).
        """
        remaining_rank = self.remaining_by_rank.get(claimed_rank, 0) - our_copies
        remaining_total = self.total_remaining - our_hand_size

        if remaining_total <= 0:
            return 1.0 if remaining_rank < claim_size else 0.0
        if claim_size > remaining_rank:
            return 1.0  # impossible claim — not enough unseen copies
        if opp_hand_size <= 0:
            return 0.0

        p_fewer = Hypergeometric.cdf(
            claim_size - 1, remaining_total, max(0, remaining_rank), opp_hand_size
        )
        return (1.0 - evidence_weight) * prior + evidence_weight * p_fewer

    def claim_plausibility(self, claimed_rank: Rank, claim_size: int,
                           our_hand_size: int, opp_hand_size: int,
                           our_copies: int = 0) -> float:
        """P(the opponent CANNOT disprove OUR claim of claim_size copies).

        The opponent can prove a bluff iff the unrevealed copies they see
        (4 − revealed − their own copies) are fewer than claim_size, i.e.
        iff their_copies > remaining_rank − claim_size.
        Used offensively: score candidate bluffs by how likely they survive.
        """
        k = self.remaining_by_rank.get(claimed_rank, 0) - claim_size
        pool_k = self.remaining_by_rank.get(claimed_rank, 0) - our_copies
        remaining_total = self.total_remaining - our_hand_size

        if k < 0:
            return 0.0  # claim exceeds all unseen copies — always disprovable
        if remaining_total <= 0 or opp_hand_size <= 0:
            return 1.0 if k >= pool_k or pool_k >= claim_size else 0.5
        if k >= pool_k:
            return 1.0  # our own copies make the claim unimpeachable
        return Hypergeometric.cdf(
            k, remaining_total, max(0, pool_k), opp_hand_size
        )


class BayesianBot(BotInterface):
    """Bayesian opponent model + card counting + offensive bluffing."""

    def __init__(self):
        self.model = OpponentModel()
        self.counter = CardCounter()
        self.bluff_tracker = BluffTracker()
        self.bluff_threshold = 0.55
        self.call_threshold = 0.55
        self.player_id: Optional[int] = None

    def reset(self):
        self.counter = CardCounter()
        # Don't reset model — persists across games

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        rank_counts: Dict[Rank, int] = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

        call_freq = self.model.estimate_call_frequency()
        hand_size = len(hand)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        pile_size = game_state.get("pile_size", 0)

        # Decide whether to bluff this turn
        should_bluff = self.bluff_tracker.should_bluff(
            hand_size, opp_hand_size, pile_size, call_freq
        )

        if should_bluff:
            return self._bluff_play(hand, rank_counts, game_state)
        else:
            return self._honest_play(hand, rank_counts, game_state)

    def _bluff_play(self, hand: List[Card], rank_counts: Dict[Rank, int],
                    game_state: dict) -> Tuple[List[Card], Rank]:
        """Execute a bluff play — claim cards are a rank we don't have (or under-represent)."""
        hand_size = len(hand)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        call_freq = self.model.estimate_call_frequency()

        best_score = -1
        best_rank = None
        best_cards: List[Card] = []

        for rank in Rank:
            have = rank_counts.get(rank, 0)
            remaining = self.counter.remaining_by_rank.get(rank, 0)

            for claim_size in range(1, min(4, hand_size) + 1):
                if claim_size > hand_size:
                    continue
                if claim_size <= have:
                    continue  # This would be honest, skip in bluff mode

                # P(opponent can't disprove this claim)
                p_undetected = self.counter.claim_plausibility(
                    rank, claim_size, hand_size, opp_hand_size, have
                )
                p_get_away = p_undetected * (1.0 - call_freq)

                # Bonus: ranks with more remaining cards are harder to detect
                # (opponent can't be sure we don't have them)
                remaining_bonus = min(0.2, remaining * 0.05)

                # Bonus: dumping more cards is better
                dump_bonus = claim_size * 0.05

                # Penalty: larger claims are riskier
                size_penalty = 1.0
                if claim_size >= 3:
                    size_penalty = 0.8
                if claim_size == 4:
                    size_penalty = 0.65

                score = (p_get_away + remaining_bonus + dump_bonus) * size_penalty

                if score > best_score:
                    best_score = score
                    best_rank = rank
                    best_cards = hand[:claim_size]

        assert best_rank is not None
        return (best_cards, best_rank)

    def _honest_play(self, hand: List[Card], rank_counts: Dict[Rank, int],
                     game_state: dict) -> Tuple[List[Card], Rank]:
        """Play honest — claim cards we actually have."""
        hand_size = len(hand)

        best_score = -1
        best_rank = None
        best_cards: List[Card] = []

        for rank in Rank:
            have = rank_counts.get(rank, 0)
            if have == 0:
                continue

            for claim_size in range(1, min(have, 4) + 1):
                if claim_size > hand_size:
                    continue

                score = claim_size * 0.1
                remaining = self.counter.remaining_by_rank.get(rank, 0)
                if remaining <= 1:
                    score += 0.2

                if score > best_score:
                    best_score = score
                    best_rank = rank
                    best_cards = [c for c in hand if c.rank == rank][:claim_size]

        assert best_rank is not None
        return (best_cards, best_rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        hand_size = game_state.get("hand_size", 10)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        pile_size = game_state.get("pile_size", 0)
        claimed_rank = last_action.claimed_rank
        claim_size = len(last_action.cards_played)

        # Card counting: P(bluff | pool model)
        # Uses the shared pool model from base.py which self-heals via
        # pending_claims — unlike the internal CardCounter whose monotone
        # remaining_by_rank permanently loses track after pile recycling.
        p_bluff_counting = pool_bluff_prob(
            game_state.get("hand", []),
            game_state.get("pending_claims", {}),
            last_action,
            opp_hand_size,
        )

        # Opponent model: P(bluff | opponent behavior)
        p_bluff_model = self.model.estimate_bluff_probability(
            opp_hand_size, claimed_rank, claim_size
        )

        # Adaptive weighting: trust model more as we get more data
        model_weight = min(1.0, self.model.total_actions_observed / 15)
        counting_weight = 1.0 - model_weight

        p_bluff = (p_bluff_counting * counting_weight +
                   p_bluff_model * model_weight)

        # Risk/reward adjustments
        pile_bonus = min(0.15, pile_size * 0.01)
        hand_bonus = 0.0
        if hand_size <= 5:
            hand_bonus = 0.1
        if hand_size <= 3:
            hand_bonus = 0.2

        threshold = max(0.50, self.call_threshold - pile_bonus - hand_bonus)

        return p_bluff > threshold

    def observe_action(self, action: Action, opponent_hand_size: int):
        is_own = (self.player_id is not None and action.player == self.player_id)

        # Update opponent model ONLY with opponent's actions.
        # Feeding own plays into the model contaminates overall_bluff
        # with self-bluff data, preventing convergence to 0% vs honest bots.
        if not is_own:
            self.model.observe_action(action, opponent_hand_size)

        # Update card counter with all KNOWN cards:
        # - Called actions: cards are revealed regardless of who played
        # - Own plays: we know what we played (even bluffs)
        # - Honest plays: cards match the claimed rank
        if action.bluff_called or is_own or not action.was_bluff:
            self.counter.update_with_play(action.cards_played)

        # Track our own bluff outcomes
        if is_own and action.was_bluff:
            self.bluff_tracker.record_bluff(action.bluff_called)

    def save(self, path: str):
        data = {
            "model": self.model.to_dict(),
            "bluff_tracker": self.bluff_tracker.to_dict(),
            "bluff_threshold": self.bluff_threshold,
            "call_threshold": self.call_threshold,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.model = OpponentModel.from_dict(data["model"])
        self.bluff_tracker = BluffTracker.from_dict(data.get("bluff_tracker", {}))
        self.bluff_threshold = data.get("bluff_threshold", 0.55)
        self.call_threshold = data.get("call_threshold", 0.50)
