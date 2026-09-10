"""L-trace: instrument HonestBot's bluff_probability() inputs at every decision.

Resolves WHY HonestBot passes vs honest opponents despite the (pre-v2-fix)
likelihood-inflation hypothesis: records the exact L value the bot sees at
every response decision, bucketed by engine turn, with pass/call decisions.

FINDINGS (10 games, seed 7, pre-bluff_probability-v2-fix, vs DumpBot):
  meanL = 0.267, medianL = 0.290, P(L>0.6) = 0.0% over 165 decisions.
  Voluntary call rate 0.0% — Honest under-calls vs honest play; the old
  always-call behavior was claim-SIZE-driven (3-4 card claims push L to
  0.75-0.99), not uniform. Draw pile exhausts ~turn 24 (can_pass flips).
"""
import sys, os, random, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState
from bots.base import build_game_state, bluff_probability, pending_claims_from_actions
from bots.honest_bot import HonestBot
from verification.probe import DumpBot


def trace_honest_decisions(games=10, max_turns=300, seed=7):
    random.seed(seed)
    rows = []  # (engine_turn, L, called, can_pass)
    for g in range(games):
        game = GameState(num_players=2)
        game.deal(14)
        honest = HonestBot(); setattr(honest, "player_id", 1)
        dump = DumpBot(); setattr(dump, "player_id", 0)
        turn = 0
        while not game.game_over and turn < max_turns:
            current = game.current_player
            other = 1 - current
            hand = game.get_hand(current)
            bot = honest if current == 1 else dump
            cards, rank = bot.decide_play(
                hand=hand.cards,
                game_state=build_game_state(
                    hand_size=hand.size(),
                    opponent_hand_size=game.get_hand(other).size(),
                    pile_size=game.get_pile_size(),
                    draw_pile_size=len(game.draw_pile),
                    turn_number=turn,
                    last_action=game.actions[-1] if game.actions else None,
                    cards_played=game.get_cards_played(),
                    actions=game.actions))
            if not cards:
                turn += 1; continue
            ok, _ = game.play_cards(current, cards, rank)
            if not ok:
                turn += 1; continue
            last_action = game.actions[-1]
            # record Honest's decision inputs when HONEST is the responder
            called = False
            can_pass = len(game.draw_pile) > 0
            if other == 1:
                pending = pending_claims_from_actions(game.actions)
                L = bluff_probability(game.get_hand(1).cards, pending,
                                      last_action, game.get_hand(0).size())
                called = honest.decide_call(
                    last_action,
                    build_game_state(
                        hand_size=game.get_hand(1).size(),
                        opponent_hand_size=game.get_hand(0).size(),
                        pile_size=game.get_pile_size(),
                        draw_pile_size=len(game.draw_pile),
                        turn_number=turn, last_action=last_action,
                        cards_played=game.get_cards_played(),
                        hand=game.get_hand(1).cards,
                        actions=game.actions))
                rows.append((game.turn_count, L, called, can_pass))
            # resolve response (mirrors probe driver: forced call when pile empty)
            should_call = (called if other == 1 else False)
            if other == 1 and not can_pass:
                should_call = True
            if should_call:
                game.call_bluff(other)
            else:
                game.pass_turn(passer=other)
            turn += 1
    # report
    print(f"\nL-trace: {len(rows)} Honest decisions across {games} games")
    print(f"{'turn':>5} {'n':>4} {'meanL':>6} {'pL>0.6':>7} {'call%':>6} {'can_pass%':>9}")
    by_turn = {}
    for t, L, c, fp in rows:
        by_turn.setdefault(min(t, 60), []).append((L, c, fp))
    for t in sorted(by_turn):
        vals = by_turn[t]
        Ls = [v[0] for v in vals]
        cs = [v[1] for v in vals]
        fps = [v[2] for v in vals]
        print(f"{t:>5} {len(vals):>4} {statistics.mean(Ls):>6.3f} "
              f"{sum(l > 0.6 for l in Ls)/len(Ls):>7.1%} "
              f"{sum(cs)/len(cs):>6.1%} {sum(fps)/len(fps):>9.1%}")
    Ls = [r[1] for r in rows]
    print(f"\noverall: meanL={statistics.mean(Ls):.3f}  "
          f"medianL={statistics.median(Ls):.3f}  "
          f"P(L>0.6)={sum(l > 0.6 for l in Ls)/len(Ls):.1%}  "
          f"voluntary-call-rate={sum(1 for r in rows if r[2] and r[3])/max(1,len(rows)):.1%}")


if __name__ == "__main__":
    trace_honest_decisions()
