"""Verification probe: is 'NN 0% vs Honest' an engine flaw or bot degeneracy?

Drives games directly through game.py per game-rules.md §2/§4 (same contract
as test_bots.py, logger-free), and adds:
  1. Card-conservation invariant (h0+h1+pile+draw == 52) after every mutation.
  2. 200-game matchups, seeded, seats swapped (larger samples).
  3. DumpBot / SmartDump / SmartDump2 / SmartDump3 — theory probes testing
     beatability of HonestBot under different bluffing policies.
  4. HonestEV — HonestBot with corrected calling: empirical per-opponent bluff
     rate theta_hat (Laplace) + Yeung-style payoff threshold theta > (P-1)/P,
     plus the impossible-claim corner. Tests the ADR fix direction.

Falsifiable predictions (from the 2026-09-10 orchestrator analysis):
  P1 conservation holds everywhere                      -> engine is sound
  P2 honest-play strategies cannot beat Honest          -> draw-lock structure
  P4 Honest beats Random ~100% (Random bluffs into      -> 100% vs Random is
     right calls, wrong-calls Honest)                      real, not a bug
  P6 4-card junk bluffs self-destruct (impossible-      -> corner works
     claim corner -> rightly called every time)
  P8 sized honest dumps (3-4 when held) still draw-lock -> criterion is
                                                           structural, not
                                                           policy-fixable

RESULTS (200 games/matchup, seed 42, pre-bluff_probability-v2-fix):
  Honest vs Random        200-0-0   | Honest vs CardCount  0-0-200
  Honest vs Bayesian      170-0-30  | Honest vs Honest     0-0-200
  DumpBot vs Honest       0-0-200   | SmartDump vs Honest  0-197-3
  SmartDump2 vs Honest    0-14-186  | SmartDump3 vs Honest 0-0-200
All conservation checks passed. See verification/__init__.py for the full
findings digest and AGENT_CHAT.md 15:35 (2026-09-10).
"""
import sys, os, random, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState
from cards import Rank
from bots.base import build_game_state, bluff_probability, BotInterface
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot


# ---------------------------------------------------------------- probe bots
class DumpBot(BotInterface):
    """Theory probe vs always-callers: honest dumps, never calls."""
    def reset(self): pass
    def decide_play(self, hand, game_state):
        if not hand:
            return ([], Rank.TWO)
        counts = {}
        for c in hand:
            counts[c.rank] = counts.get(c.rank, 0) + 1
        best = max(counts.keys(), key=lambda r: counts[r])
        n = min(counts[best], 4)
        return ([c for c in hand if c.rank == best][:n], best)
    def decide_call(self, last_action, game_state):
        return False
    def observe_action(self, action, opponent_hand_size): pass
    def save(self, path): pass
    def load(self, path): pass


class SmartDump(BotInterface):
    """Bluff-dump exploit attempt: dump ANY 4 cards claiming 2s, never call.

    FALSIFIED: the impossible-claim corner (pool_k < claim_size -> L=1.0)
    rightly catches near-every 4-card junk claim -> self-destructs.
    """
    def reset(self): pass
    def decide_play(self, hand, game_state):
        if not hand:
            return ([], Rank.TWO)
        n = min(4, len(hand))
        return (list(hand)[:n], Rank.TWO)   # claim 2s regardless — pure dump
    def decide_call(self, last_action, game_state):
        return False
    def observe_action(self, action, opponent_hand_size): pass
    def save(self, path): pass
    def load(self, path): pass


class SmartDump2(BotInterface):
    """Plausible-bluff dumper: <=2 cards, claim one of the dumped ranks.

    Avoids the impossible-claim corner, but still draw-locks (no baiting).
    """
    def reset(self): pass
    def decide_play(self, hand, game_state):
        if not hand:
            return ([], Rank.TWO)
        n = min(2, len(hand))
        cards = list(hand)[:n]
        return (cards, cards[0].rank)   # claim a rank actually present
    def decide_call(self, last_action, game_state):
        return False
    def observe_action(self, action, opponent_hand_size): pass
    def save(self, path): pass
    def load(self, path): pass


class SmartDump3(BotInterface):
    """Sized honest dumps: 3-4 cards of best rank when held, else 2. Never call.

    3-4 card honest claims get called (L~0.85-0.99 pre-fix) -> wrong call ->
    full claim shed; 2-card claims (L~0.4) get passed. Either way our hand
    shrinks — but the race is symmetric, so it still draw-locks.
    """
    def reset(self): pass
    def decide_play(self, hand, game_state):
        if not hand:
            return ([], Rank.TWO)
        counts = {}
        for c in hand:
            counts[c.rank] = counts.get(c.rank, 0) + 1
        best = max(counts.keys(), key=lambda r: counts[r])
        n = min(counts[best], 4) if counts[best] >= 3 else min(counts[best], 2)
        return ([c for c in hand if c.rank == best][:n], best)
    def decide_call(self, last_action, game_state):
        return False
    def observe_action(self, action, opponent_hand_size): pass
    def save(self, path): pass
    def load(self, path): pass


class HonestEV(HonestBot):
    """HonestBot + corrected calling: empirical theta_hat, EV threshold.

    Demonstrated that prior-blending the likelihood does NOT fix calling
    (HonestEV ≡ Honest empirically vs honest opponents): the binding
    constraint is the draw-clock + forced-call structure, not the threshold.
    """
    def reset(self):
        self._n = 0
        self._bluffs = 0

    def decide_call(self, last_action, game_state):
        L = bluff_probability(
            game_state.get("hand", []),
            game_state.get("pending_claims", {}),
            last_action,
            game_state.get("opponent_hand_size", 14),
        )
        if L >= 1.0:          # impossible claim — always bluff
            return True
        theta = (self._bluffs + 1) / (self._n + 4)   # Laplace-smoothed rate
        P = max(1, game_state.get("pile_size", 1))
        return theta > (P - 1) / P                  # payoff-ratio threshold

    def observe_action(self, action, opponent_hand_size):
        opp = 1 - getattr(self, "player_id", 0)
        if action.player == opp and action.bluff_called:
            self._n += 1
            self._bluffs += 1 if action.was_bluff else 0

    def save(self, path): pass
    def load(self, path): pass


# ------------------------------------------------------------------- driver
def check_conservation(game, where):
    total = (game.hands[0].size() + game.hands[1].size()
             + len(game.pile) + len(game.draw_pile))
    if total != 52:
        raise RuntimeError(f"CONSERVATION VIOLATION ({where}): {total} != 52 "
                           f"(h0={game.hands[0].size()} h1={game.hands[1].size()} "
                           f"pile={len(game.pile)} draw={len(game.draw_pile)})")


def play_game(make_a, make_b, max_turns=300):
    game = GameState(num_players=2)
    game.deal(14)
    check_conservation(game, "deal")
    bots = [make_a(), make_b()]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()

    turn = 0
    calls = [0, 0]; wrong = [0, 0]; forced = [0, 0]
    plays = [0, 0]; bluffs = [0, 0]
    while not game.game_over and turn < max_turns:
        current = game.current_player
        other = 1 - current
        hand = game.get_hand(current)
        cards, rank = bots[current].decide_play(
            hand=hand.cards,
            game_state=build_game_state(
                hand_size=hand.size(),
                opponent_hand_size=game.get_hand(other).size(),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=turn,
                last_action=game.actions[-1] if game.actions else None,
                cards_played=game.get_cards_played(),
                actions=game.actions,
            ),
        )
        if not cards:
            turn += 1
            continue
        ok, _ = game.play_cards(current, cards, rank)
        if not ok:
            turn += 1
            continue
        plays[current] += 1
        bluffs[current] += 1 if game.actions[-1].was_bluff else 0
        check_conservation(game, "play")

        last_action = game.actions[-1]
        gs = build_game_state(
            hand_size=game.get_hand(other).size(),
            opponent_hand_size=game.get_hand(current).size(),
            pile_size=game.get_pile_size(),
            draw_pile_size=len(game.draw_pile),
            turn_number=turn,
            last_action=last_action,
            cards_played=game.get_cards_played(),
            hand=game.get_hand(other).cards,
            actions=game.actions,
        )
        if game.can_call_bluff():
            if not game.can_pass():
                should_call = True
                forced[other] += 1
            else:
                should_call = bots[other].decide_call(last_action, gs)
        else:
            should_call = False

        if should_call:
            ok2, _, action = game.call_bluff(other)
            if ok2 and action:
                calls[other] += 1
                if not action.caller_was_right:
                    wrong[other] += 1
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            # pass (draws 1); if draw pile empty this is a no-draw pass —
            # the forced-call branch above prevents reaching here illegally
            game.pass_turn(passer=other)
        check_conservation(game, "respond")
        turn += 1

    winner = game.winner if game.game_over else None
    return {
        "winner": winner, "turns": game.turn_count,
        "end_hands": (game.hands[0].size(), game.hands[1].size()),
        "calls": calls, "wrong": wrong, "forced": forced,
        "plays": plays, "bluffs": bluffs,
    }


def run_matchup(name, make_a, make_b, games=200, seed=42):
    random.seed(seed)
    w = l = d = 0
    agg = {"calls": [0, 0], "wrong": [0, 0], "forced": [0, 0],
           "plays": [0, 0], "bluffs": [0, 0]}
    turns_list, hands = [], []
    for i in range(games):
        a_first = (i % 2 == 0)
        r = play_game(make_a if a_first else make_b,
                      make_b if a_first else make_a)
        seat_a = 0 if a_first else 1
        if r["winner"] is None:
            d += 1
        elif r["winner"] == seat_a:
            w += 1
        else:
            l += 1
        turns_list.append(r["turns"])
        hands.append(r["end_hands"])
        # seat-correct aggregation: seat s belongs to A iff s == seat_a
        for k in agg:
            for s in (0, 1):
                agg[k][0 if s == seat_a else 1] += r[k][s]
    n = games
    ci = 1.96 * ((w / n) * (1 - w / n) / n) ** 0.5
    print(f"\n=== {name} — {games} games (seed {seed}, seats swapped) ===")
    print(f"  A(win/loss/draw): {w}-{l}-{d}  | A win rate {w/n:.1%} ± {ci:.1%}")
    print(f"  avg turns: {statistics.mean(turns_list):.1f}  "
          f"median: {statistics.median(turns_list):.0f}")
    print(f"  avg end hands (A,B): "
          f"({statistics.mean(h[0] for h in hands):.1f}, "
          f"{statistics.mean(h[1] for h in hands):.1f})")
    for s, tag in ((0, "A"), (1, "B")):
        pl = max(1, agg["plays"][s]); cl = max(1, agg["calls"][s])
        print(f"  {tag}: plays {agg['plays'][s]}  bluffs {agg['bluffs'][s]/pl:.1%}  "
              f"calls {agg['calls'][s]} (wrong {agg['wrong'][s]/cl:.1%}, "
              f"forced {agg['forced'][s]})")
    return w, l, d


def make(cls):
    return lambda: cls()


if __name__ == "__main__":
    print(__doc__)
    run_matchup("Honest vs Random", make(HonestBot), make(RandomBot))
    run_matchup("Honest vs CardCount", make(HonestBot), make(CardCountBot))
    run_matchup("Honest vs Bayesian", make(HonestBot), make(BayesianBot))
    run_matchup("Honest vs Honest", make(HonestBot), make(HonestBot))
    run_matchup("DumpBot vs Honest   [P2: honest-play beatability]", make(DumpBot), make(HonestBot))
    run_matchup("SmartDump vs Honest [P6: bluff-dump exploits]", make(SmartDump), make(HonestBot))
    run_matchup("SmartDump2 vs Honest [P7: plausible bluffs]", make(SmartDump2), make(HonestBot))
    run_matchup("SmartDump3 vs Honest [P8: sized dumps]", make(SmartDump3), make(HonestBot))
    run_matchup("HonestEV vs Honest [P5: EV-fix vs honest]", make(HonestEV), make(HonestBot))
    run_matchup("HonestEV vs DumpBot [P5 sanity]", make(HonestEV), make(DumpBot))
    run_matchup("CardCount vs DumpBot", make(CardCountBot), make(DumpBot))
    print("\nAll conservation checks passed (52 cards at every step).")
