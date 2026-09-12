"""
NN-floor × Thompson sweep on AcademicBeastBot — the Bayesian-vs-NN balance
frontier the A23 study left coarse (w_nn=0 vs hybrids only).

Grid: nn_floor in {0.0, 0.15, 0.30} x thompson_sampling in {True, False} =
6 conditions x 4 rule-bot opponents (Honest/Bayesian/CardCount/Random) x
N=100/matchup, strict 50/50 seats, fixed seeds. ~2,400 games.

Rationale: nn_floor is the live knob controlling NN-vs-Bayesian balance
(risk_aversion is stored-but-unused — verified by grep, NOT swept);
Thompson interaction with the floor is unmeasured. decay_tau already
swept (optimal 8.0); schedule fixed exponential (current default).

Harness mirrors test_bots.py exactly: observe_action on BOTH bots for every
resolved call (voluntary + forced), fabricated pass_act on passes (public
info only). Wilson CIs in output.

Run:  python3 experiments/nn_floor_sweep.py [--smoke]
"""
import sys, os, argparse, json, random, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import GameState, Action
from cards import Rank
from bots.base import build_game_state
from bots.academic_beast_bot import AcademicBeastBot
from bots.honest_bot import HonestBot
from bots.bayesian_bot import BayesianBot
from bots.cardcount_bot import CardCountBot
from bots.random_bot import RandomBot


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, c - m), min(1.0, c + m)


def play_game(make_measured, make_opponent, measured_seat, max_turns=100):
    game = GameState(num_players=2)
    game.deal(14)
    bots = [make_measured(), make_opponent()]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()
    turn = 0
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
            should_call = (not game.can_pass()) or bots[other].decide_call(last_action, gs)
        else:
            should_call = False
        if should_call:
            ok2, _, action = game.call_bluff(other)
            if ok2 and action:
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            if game.can_pass():
                okp, _ = game.pass_turn(passer=other)
                if okp:
                    pa = Action(player=other, cards_played=[], claimed_rank=Rank.TWO,
                                was_bluff=False, bluff_called=False,
                                caller_was_right=False,
                                pile_size_before=game.get_pile_size())
                    bots[current].observe_action(pa, game.get_hand(other).size())
                    bots[other].observe_action(pa, game.get_hand(current).size())
        turn += 1
    return game.winner


OPPONENTS = {
    "Honest": HonestBot,
    "Bayesian": BayesianBot,
    "CardCount": CardCountBot,
    "Random": RandomBot,
}

CONDITIONS = {
    "floor0.00_thompsonON": dict(nn_floor=0.00, thompson_sampling=True),
    "floor0.00_thompsonOFF": dict(nn_floor=0.00, thompson_sampling=False),
    "floor0.15_thompsonON": dict(nn_floor=0.15, thompson_sampling=True),
    "floor0.15_thompsonOFF": dict(nn_floor=0.15, thompson_sampling=False),
    "floor0.30_thompsonON": dict(nn_floor=0.30, thompson_sampling=True),
    "floor0.30_thompsonOFF": dict(nn_floor=0.30, thompson_sampling=False),
}


def run(n_games: int, seed: int):
    random.seed(seed)
    out = {}
    for cname, params in CONDITIONS.items():
        out[cname] = {}
        tw = tl = td = 0
        for oname, ocls in OPPONENTS.items():
            w = l = d = 0
            for g in range(n_games):
                measured_seat = g % 2
                if measured_seat == 0:
                    winner = play_game(
                        lambda p=params: AcademicBeastBot(
                            nn_floor=p["nn_floor"],
                            thompson_sampling=p["thompson_sampling"]), ocls, 0)
                else:
                    # swap slots: opponent sits seat 0
                    winner = play_measured_swapped(
                        lambda p=params: AcademicBeastBot(
                            nn_floor=p["nn_floor"],
                            thompson_sampling=p["thompson_sampling"]), ocls)
                if winner is None:
                    d += 1
                elif winner == measured_seat:
                    w += 1
                else:
                    l += 1
            lo, hi = wilson(w, n_games)
            out[cname][oname] = {"w": w, "l": l, "d": d,
                                 "win_rate": 100.0 * w / n_games,
                                 "w_ci": [100.0 * lo, 100.0 * hi]}
            tw += w; tl += l; td += d
            print(f"  {cname:>24} vs {oname:<10}: {w:>3}W-{l:<3}-{d:<3} "
                  f"| win {100*w/n_games:5.1f}% CI[{100*lo:.1f},{100*hi:.1f}]",
                  flush=True)
        lo, hi = wilson(tw, tw + tl + td)
        print(f"  [{cname}] TOTAL {tw}W-{tl}L-{td}D win {100*tw/(tw+tl+td):.1f}% "
              f"CI[{100*lo:.1f},{100*hi:.1f}]", flush=True)
    return out


def play_measured_swapped(make_measured, opp_cls):
    # opponent in seat 0, measured bot in seat 1; returns winner SEAT
    game = GameState(num_players=2)
    game.deal(14)
    bots = [opp_cls(), make_measured()]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()
    turn = 0
    while not game.game_over and turn < 100:
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
            should_call = (not game.can_pass()) or bots[other].decide_call(last_action, gs)
        else:
            should_call = False
        if should_call:
            ok2, _, action = game.call_bluff(other)
            if ok2 and action:
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            if game.can_pass():
                okp, _ = game.pass_turn(passer=other)
                if okp:
                    pa = Action(player=other, cards_played=[], claimed_rank=Rank.TWO,
                                was_bluff=False, bluff_called=False,
                                caller_was_right=False,
                                pile_size_before=game.get_pile_size())
                    bots[current].observe_action(pa, game.get_hand(other).size())
                    bots[other].observe_action(pa, game.get_hand(current).size())
        turn += 1
    return game.winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n = 2 if args.smoke else 100
    print(f"NN-floor x Thompson sweep — N={n}/matchup x 4 opponents x 6 conditions, "
          f"seed 20260915", flush=True)
    random.seed(20260915)
    import time
    t0 = time.time()
    out = run(n, 20260915)
    with open("data/nn_floor_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] saved data/nn_floor_sweep.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
