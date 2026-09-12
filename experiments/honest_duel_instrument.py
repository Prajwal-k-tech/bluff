"""
Comprehensive instrumented Beast-vs-Honest duel (T7c diagnostic).

One harness, full picture, seeded. Per game + aggregate:
- winner, turns played
- Beast bluffs (count + claim-size distribution)
- Honest calls split: on Beast bluffs vs on Beast honest plays
- pile absorption by cause: Beast wrong-calls vs Beast caught-bluffs
  (vs Honest every Beast call is wrong — Honest never bluffs)
- end hand sizes, Beast call rate, archetype end-state distribution

Run: python3 experiments/honest_duel_instrument.py [--games N]  (default 30)
"""
import sys, os, argparse, json, random, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import GameState
from bots.base import build_game_state
from bots.academic_beast_bot import AcademicBeastBot
from bots.honest_bot import HonestBot


def play_one(game_id):
    game = GameState(num_players=2)
    game.deal(14)
    beast, honest = AcademicBeastBot(), HonestBot()
    beast.player_id = 0
    beast.reset(); honest.reset()
    rec = {"bluffs": [], "honest_plays": 0,
           "honest_calls_on_bluff": 0, "honest_calls_on_honest": 0,
           "beast_wrong_call_cost": 0, "beast_caught_bluff_cost": 0,
           "beast_calls": 0, "beast_call_opps": 0}
    turn = 0
    while not game.game_over and turn < 100:
        cur = game.current_player; oth = 1 - cur
        hand = game.get_hand(cur)
        bots = [beast, honest]
        cards, rank = bots[cur].decide_play(
            hand=hand.cards,
            game_state=build_game_state(
                hand_size=hand.size(),
                opponent_hand_size=game.get_hand(oth).size(),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=turn,
                last_action=game.actions[-1] if game.actions else None,
                cards_played=game.get_cards_played(),
                actions=game.actions))
        if not cards:
            turn += 1; continue
        ok, _ = game.play_cards(cur, cards, rank)
        if not ok:
            turn += 1; continue
        la = game.actions[-1]
        is_bluff = not all(c.rank == rank for c in cards)
        if cur == 0:
            if is_bluff: rec["bluffs"].append(len(cards))
            else: rec["honest_plays"] += 1
        gs = build_game_state(
            hand_size=game.get_hand(oth).size(),
            opponent_hand_size=game.get_hand(cur).size(),
            pile_size=game.get_pile_size(),
            draw_pile_size=len(game.draw_pile),
            turn_number=turn, last_action=la,
            cards_played=game.get_cards_played(),
            hand=game.get_hand(oth).cards, actions=game.actions)
        can_call = game.can_call_bluff()
        if can_call and oth == 0: rec["beast_call_opps"] += 1
        sc = ((not game.can_pass()) or bots[oth].decide_call(la, gs)) if can_call else False
        if sc:
            if oth == 0: rec["beast_calls"] += 1
            pile_before = game.get_pile_size()
            ok2, _, action = game.call_bluff(oth)
            if ok2 and action:
                if cur == 0 and oth == 1:  # Honest calls Beast
                    if is_bluff:
                        rec["honest_calls_on_bluff"] += 1
                        rec["beast_caught_bluff_cost"] += pile_before
                    else:
                        rec["honest_calls_on_honest"] += 1
                        # Honest wrong -> Honest absorbs; Beast gains nothing directly
                if cur == 1 and oth == 0:  # Beast calls Honest (always wrong)
                    rec["beast_wrong_call_cost"] += pile_before
                bots[cur].observe_action(action, game.get_hand(oth).size())
                bots[oth].observe_action(action, game.get_hand(cur).size())
        elif game.can_pass():
            game.pass_turn(passer=oth)
        turn += 1
    top, conf = beast.classifier.top_archetype(
        beast.opp_bluffs, beast.opp_honest, beast.opp_calls, beast.opp_passes)
    return {"winner": game.winner, "turns": turn,
            "end": (game.get_hand(0).size(), game.get_hand(1).size()),
            "arch": (top, round(conf, 2)), **rec}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--games", type=int, default=30)
    n = ap.parse_args().games
    random.seed(20260912)
    games = [play_one(i) for i in range(n)]
    agg = {"n": n, "wins": collections.Counter(g["winner"] for g in games),
           "bluffs_total": sum(len(g["bluffs"]) for g in games),
           "bluff_sizes": collections.Counter(s for g in games for s in g["bluffs"]),
           "honest_plays": sum(g["honest_plays"] for g in games),
           "honest_calls_on_bluff": sum(g["honest_calls_on_bluff"] for g in games),
           "honest_calls_on_honest": sum(g["honest_calls_on_honest"] for g in games),
           "beast_wrong_call_cost": sum(g["beast_wrong_call_cost"] for g in games),
           "beast_caught_bluff_cost": sum(g["beast_caught_bluff_cost"] for g in games),
           "beast_calls": sum(g["beast_calls"] for g in games),
           "beast_call_opps": sum(g["beast_call_opps"] for g in games),
           "arch_end": {"%s|%.2f" % (t, c): v
                        for (t, c), v in
                        collections.Counter(g["arch"] for g in games).items()}}
    print(json.dumps(agg, default=str, indent=1))
    with open("data/honest_duel.json", "w") as f:
        json.dump({"agg": agg, "games": games}, f, default=str)


if __name__ == "__main__":
    main()
