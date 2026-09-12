"""Shed/call diagnostic: WHERE does each bot bleed vs Honest/Bayesian?

Drives games directly through game.py (same contract as test_bots.py).
For every ordered matchup, reports per-bot:
  shed/turn  — cards shed per own play turn (shedding race speed)
  call%       — fraction of responses that are calls
  wrong%      — fraction of calls where the CALLER takes the pile
  took/game   — avg cards absorbed via pile takes per game
  W/L/D, avg length

Diagnostic (N=50, not citable) — finds the leak; citable runs follow.
Run:  python3 experiments/shed_call_diagnostic.py
"""
import sys, os, random, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState
from bots.base import build_game_state
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from bots.pure_nn_bot import PureNNBot
from bots.hybrid_bot import HybridBot


def play_instrumented(bot_a, bot_b, seed):
    random.seed(seed)
    game = GameState(num_players=2)
    game.deal(14)
    bots = [bot_a, bot_b]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()
    # alternate first player by seed parity (seat fairness)
    game.current_player = seed % 2

    s = [{"shed": 0, "plays": 0, "calls": 0, "wrong": 0, "took": 0, "resp": 0}
         for _ in range(2)]
    turn = 0
    while not game.game_over and turn < 300:
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
        s[current]["shed"] += len(cards)
        s[current]["plays"] += 1

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
        s[other]["resp"] += 1
        if game.can_call_bluff():
            should_call = (not game.can_pass()) or bots[other].decide_call(last_action, gs)
        else:
            should_call = False
        if should_call:
            before = game.get_hand(other).size()
            ok2, _, action = game.call_bluff(other)
            if ok2 and action:
                s[other]["calls"] += 1
                after = game.get_hand(other).size()
                if after > before:
                    s[other]["wrong"] += 1
                    s[other]["took"] += after - before
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            if game.can_pass():
                game.pass_turn(passer=other)
        turn += 1

    return {"winner": game.winner, "turns": game.turn_count,
            "end": (game.hands[0].size(), game.hands[1].size()), "stats": s}


def run_matchup(name_a, make_a, name_b, make_b, games=50, seed=7):
    agg_w = agg_l = agg_d = 0
    agg = {"A": {"shed": 0, "plays": 0, "calls": 0, "wrong": 0, "took": 0, "resp": 0},
           "B": {"shed": 0, "plays": 0, "calls": 0, "wrong": 0, "took": 0, "resp": 0}}
    turns = []
    for i in range(games):
        # seat swap: A sits seat (i%2==0 ? 0 : 1)
        if i % 2 == 0:
            r = play_instrumented(make_a(), make_b(), seed + i)
            a_stats, b_stats, seat_a = r["stats"][0], r["stats"][1], 0
        else:
            r = play_instrumented(make_b(), make_a(), seed + i)
            b_stats, a_stats, seat_a = r["stats"][0], r["stats"][1], 1

        def acc(dst, src):
            for k in dst:
                dst[k] += src[k]
        acc(agg["A"], a_stats)
        acc(agg["B"], b_stats)
        turns.append(r["turns"])
        if r["winner"] is None:
            agg_d += 1
        elif r["winner"] == seat_a:
            agg_w += 1
        else:
            agg_l += 1

    def line(tag, st):
        shed_t = st["shed"] / max(1, st["plays"])
        call_r = st["calls"] / max(1, st["resp"])
        wrong_r = st["wrong"] / max(1, st["calls"])
        took_g = st["took"] / games
        return (f"  {tag}: shed/turn {shed_t:.2f} | call {call_r:.0%} "
                f"({st['calls']}) | wrong {wrong_r:.0%} | took/game {took_g:.1f}")
    print(f"\n=== {name_a} (A) vs {name_b} (B) — {games} games ===")
    print(f"  A {agg_w}-{agg_l}-{agg_d} | avg turns {statistics.mean(turns):.0f}")
    print(line("A", agg["A"]))
    print(line("B", agg["B"]))


ROSTER = {
    "Hybrid": lambda: HybridBot(),
    "PureNN-final": lambda: PureNNBot("nn/checkpoints/final.pt"),
    "PureNN-v61": lambda: PureNNBot("nn/checkpoints/v61_best.pt"),
    "Bayesian": lambda: BayesianBot(),
    "CardCount": lambda: CardCountBot(),
}
OPPS = {
    "Honest": lambda: HonestBot(),
    "Bayesian": lambda: BayesianBot(),
}

if __name__ == "__main__":
    print("Shed/call diagnostic — N=50/matchup, seats swapped (diagnostic, not citable)")
    for aname, amake in ROSTER.items():
        for oname, omake in OPPS.items():
            if aname == oname:
                continue
            run_matchup(aname, amake, oname, omake)
    print("\nConservation: engine previously verified sound (probe.py).")
