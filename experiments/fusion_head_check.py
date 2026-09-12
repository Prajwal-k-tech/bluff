"""
Fusion-head checkpoint check (T7i): which NN head is better INSIDE the
deploy stack? Compares a challenger checkpoint vs the champion (v61_best)
as AcademicBeastBot NN heads, vs Honest + Bayesian (discriminative pair),
N=100/matchup, strict 50/50 seats, fixed seed.

Training evals measure the RAW policy (reads 0% vs non-Random for
structural reasons); this measures the head where it actually ships.

Challenger is copied to /tmp first with load-retry: training writes
checkpoints non-atomically, so a direct read can catch a half-written file.

Run:  python3 experiments/fusion_head_check.py [--smoke] [--challenger PATH]
                                                           [--champion PATH]
"""
import sys, os, argparse, json, random, math, shutil, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import GameState, Action
from cards import Rank
from bots.base import build_game_state
from bots.academic_beast_bot import AcademicBeastBot
from bots.honest_bot import HonestBot
from bots.bayesian_bot import BayesianBot


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, c - m), min(1.0, c + m)


def play_game(bot0, bot1, max_turns=100):
    game = GameState(num_players=2)
    game.deal(14)
    bots = [bot0, bot1]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            try:
                b.player_id = i
            except AttributeError:
                pass
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
                actions=game.actions))
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
            turn_number=turn, last_action=last_action,
            cards_played=game.get_cards_played(),
            hand=game.get_hand(other).cards, actions=game.actions)
        if game.can_call_bluff():
            should_call = ((not game.can_pass())
                           or bots[other].decide_call(last_action, gs))
        else:
            should_call = False
        if should_call:
            ok2, _, action = game.call_bluff(other)
            if ok2 and action:
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        elif game.can_pass():
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


def load_copy(src, tag):
    """Copy-then-load with one retry (source may be mid-write)."""
    import torch
    dst = f"/tmp/fusion_{tag}.pt"
    for attempt in range(3):
        try:
            shutil.copy2(src, dst)
            torch.load(dst, map_location="cpu", weights_only=True)
            return dst
        except Exception as e:
            print(f"  [load-retry {attempt}] {e}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"could not load {src} after 3 tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--challenger", default="nn/checkpoints/v9_best.pt")
    ap.add_argument("--champion", default="nn/checkpoints/v61_best.pt")
    args = ap.parse_args()
    n = 2 if args.smoke else 100
    random.seed(20260912)
    chall = load_copy(args.challenger, "challenger")
    champ = load_copy(args.champion, "champion")
    heads = {"challenger": chall, "champion": champ}
    opps = {"Honest": HonestBot, "Bayesian": BayesianBot}
    out = {}
    for hname, hpath in heads.items():
        out[hname] = {}
        for oname, ocls in opps.items():
            w = l = d = 0
            for g in range(n):
                seat = g % 2
                if seat == 0:
                    winner = play_game(AcademicBeastBot(checkpoint_path=hpath),
                                       ocls())
                else:
                    winner = play_game(ocls(),
                                       AcademicBeastBot(checkpoint_path=hpath))
                    winner = 1 - winner if winner is not None else None
                if winner is None:
                    d += 1
                elif winner == 0:
                    w += 1
                else:
                    l += 1
            lo, hi = wilson(w, n)
            out[hname][oname] = {"w": w, "l": l, "d": d,
                                 "win_rate": 100.0 * w / n,
                                 "w_ci": [100.0 * lo, 100.0 * hi]}
            print(f"  {hname:>10} vs {oname:<10}: {w:>3}W-{l:<3}-{d:<3} "
                  f"| win {100*w/n:5.1f}% CI[{100*lo:.1f},{100*hi:.1f}]",
                  flush=True)
    if not args.smoke:
        with open("data/fusion_head_check.json", "w") as f:
            json.dump(out, f, indent=2)
        print("[OK] saved data/fusion_head_check.json", flush=True)


if __name__ == "__main__":
    main()
