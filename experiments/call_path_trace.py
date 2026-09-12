"""Call-path trace: decompose HybridBot's decide_call vs HonestBot.

For every response decision, records which branch fires:
  forced (can_pass False) / corner (counting_p >= 0.999) /
  guard-blocked (mean<0.15 & counting<0.95) / fusion-call / fusion-pass.
Also records the model posterior mean trajectory.

Run:  python3 experiments/call_path_trace.py [games]
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from game import GameState, Action
from cards import Rank
from bots.base import build_game_state
from bots.hybrid_bot import HybridBot
from bots.honest_bot import HonestBot


def trace(n_games=20, seed=99):
    random.seed(seed)
    counts = {"forced": 0, "corner": 0, "guard_blocked": 0,
              "fusion_call": 0, "fusion_pass": 0, "total_resp": 0}
    means = []
    cp_vals = []
    for g in range(n_games):
        game = GameState(num_players=2)
        game.deal(14)
        hyb, hon = HybridBot(), HonestBot()
        hyb_seat = 0 if g % 2 == 0 else 1
        hon_seat = 1 - hyb_seat
        setattr(hyb, "player_id", hyb_seat)
        setattr(hon, "player_id", hon_seat)
        hyb.reset(); hon.reset()
        game.current_player = g % 2
        turn = 0
        while not game.game_over and turn < 300:
            cur = game.current_player
            other = 1 - cur
            bot = hyb if cur == hyb_seat else hon
            hand = game.get_hand(cur)
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
                turn += 1
                continue
            ok, _ = game.play_cards(cur, cards, rank)
            if not ok:
                turn += 1
                continue
            last = game.actions[-1]
            # only trace Hybrid's responses
            if other == hyb_seat and game.can_call_bluff():
                counts["total_resp"] += 1
                if not game.can_pass():
                    counts["forced"] += 1
                    ok_f, _, act_f = game.call_bluff(other)
                    if ok_f and act_f:
                        hyb.observe_action(act_f, game.get_hand(cur).size())
                        hon.observe_action(act_f, game.get_hand(other).size())
                else:
                    h = game.get_hand(other).cards
                    our_copies = sum(1 for c in h if c.rank == last.claimed_rank)
                    cp = hyb.counter.bluff_probability(
                        last.claimed_rank, len(last.cards_played),
                        len(h), game.get_hand(cur).size(), our_copies=our_copies)
                    ob = getattr(hyb.model, "overall_bluff", None)
                    mean = ob.mean() if ob is not None else 0.3
                    means.append(mean)
                    cp_vals.append(cp)
                    if cp >= 0.999:
                        counts["corner"] += 1
                        ok_c, _, act_c = game.call_bluff(other)
                        if ok_c and act_c:
                            hyb.observe_action(act_c, game.get_hand(cur).size())
                            hon.observe_action(act_c, game.get_hand(other).size())
                    elif mean < 0.15 and cp < 0.95:
                        counts["guard_blocked"] += 1
                        ok_p, _ = game.pass_turn(passer=other)
                        if ok_p:
                            _pass_act = Action(
                                player=other, cards_played=[], claimed_rank=Rank.TWO,
                                was_bluff=False, bluff_called=False,
                                caller_was_right=False,
                                pile_size_before=game.get_pile_size())
                            hyb.observe_action(_pass_act, game.get_hand(cur).size())
                            hon.observe_action(_pass_act, game.get_hand(other).size())
                    else:
                        called = hyb.decide_call(last, build_game_state(
                            hand_size=len(h),
                            opponent_hand_size=game.get_hand(cur).size(),
                            pile_size=game.get_pile_size(),
                            draw_pile_size=len(game.draw_pile),
                            turn_number=turn, last_action=last,
                            cards_played=game.get_cards_played(),
                            hand=h, actions=game.actions))
                        if called:
                            counts["fusion_call"] += 1
                            ok_c2, _, act_c2 = game.call_bluff(other)
                            if ok_c2 and act_c2:
                                hyb.observe_action(act_c2, game.get_hand(cur).size())
                                hon.observe_action(act_c2, game.get_hand(other).size())
                        else:
                            counts["fusion_pass"] += 1
                            ok_p2, _ = game.pass_turn(passer=other)
                            if ok_p2:
                                _pass_act2 = Action(
                                    player=other, cards_played=[], claimed_rank=Rank.TWO,
                                    was_bluff=False, bluff_called=False,
                                    caller_was_right=False,
                                    pile_size_before=game.get_pile_size())
                                hyb.observe_action(_pass_act2, game.get_hand(cur).size())
                                hon.observe_action(_pass_act2, game.get_hand(other).size())
            elif game.can_call_bluff():
                # Honest's response — drive normally, untraced
                gs = build_game_state(
                    hand_size=game.get_hand(other).size(),
                    opponent_hand_size=game.get_hand(cur).size(),
                    pile_size=game.get_pile_size(),
                    draw_pile_size=len(game.draw_pile),
                    turn_number=turn, last_action=last,
                    cards_played=game.get_cards_played(),
                    hand=game.get_hand(other).cards, actions=game.actions)
                if (not game.can_pass()) or hon.decide_call(last, gs):
                    ok2, _, act = game.call_bluff(other)
                else:
                    if game.can_pass():
                        game.pass_turn(passer=other)
            else:
                if game.can_pass():
                    game.pass_turn(passer=other)
            turn += 1

    n = max(1, counts["total_resp"])
    print(f"Call-path trace — Hybrid responses vs HonestBot ({n_games} games, {n} responses):")
    for k in ("forced", "corner", "guard_blocked", "fusion_call", "fusion_pass"):
        print(f"  {k:>14}: {counts[k]:>5} ({counts[k]/n:.1%})")
    if means:
        import statistics
        print(f"  model-mean: avg {statistics.mean(means):.3f} | "
              f"P(mean<0.15) = {sum(1 for m in means if m < 0.15)/len(means):.1%}")
    if cp_vals:
        import statistics as _st
        nb = len(cp_vals)
        print(f"  counting_p: avg {sum(cp_vals)/nb:.3f} | "
              f"P(<0.50)={sum(1 for v in cp_vals if v < 0.50)/nb:.1%} | "
              f"P(<0.95)={sum(1 for v in cp_vals if v < 0.95)/nb:.1%} | "
              f"P(>=0.95)={sum(1 for v in cp_vals if v >= 0.95)/nb:.1%} | "
              f"P(>=0.999)={sum(1 for v in cp_vals if v >= 0.999)/nb:.1%}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    trace(n)
