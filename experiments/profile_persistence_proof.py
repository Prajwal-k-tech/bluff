"""
Profile-persistence proof (T8 vertical slice): does a loaded profile make
game 2+ measurably better than game 1 vs a fixed persona?

Session 1: fresh AcademicBeastBot vs HonestBot x N games (seeded).
  -> mark_session_completed(lam=1.0) -> to_dict -> file.
Roundtrip assert: counters + sessions_observed survive exactly.
Decay unit: apply_session_decay(0.7) pulls means toward fresh priors.
Session 2: from_dict -> N games vs Honest.
Verdict: voluntary calls, wrong-call cost, winners session 1 vs 2.

Run: python3 experiments/profile_persistence_proof.py [--games N]
"""
import sys, os, argparse, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import GameState, Action
from cards import Rank
from bots.base import build_game_state
from bots.academic_beast_bot import AcademicBeastBot
from bots.honest_bot import HonestBot
from bots.bayesian_bot import BayesianBot

PERSONAS = {"honest": HonestBot, "bayesian": BayesianBot}


def play_game(beast, opp, max_turns=100):
    game = GameState(num_players=2)
    game.deal(14)
    bots = [beast, opp]
    for i, b in enumerate(bots):
        try:
            b.player_id = i
        except AttributeError:
            pass
        b.reset()
    rec = {"vol_calls": 0, "wrong_cost": 0, "beast_bluffs": 0}
    turn = 0
    while not game.game_over and turn < max_turns:
        cur = game.current_player
        oth = 1 - cur
        hand = game.get_hand(cur)
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
            turn += 1
            continue
        ok, _ = game.play_cards(cur, cards, rank)
        if not ok:
            turn += 1
            continue
        la = game.actions[-1]
        if cur == 0 and not all(c.rank == rank for c in cards):
            rec["beast_bluffs"] += 1
        gs = build_game_state(
            hand_size=game.get_hand(oth).size(),
            opponent_hand_size=game.get_hand(cur).size(),
            pile_size=game.get_pile_size(),
            draw_pile_size=len(game.draw_pile),
            turn_number=turn, last_action=la,
            cards_played=game.get_cards_played(),
            hand=game.get_hand(oth).cards, actions=game.actions)
        forced = not game.can_pass()
        if game.can_call_bluff():
            called = forced or bots[oth].decide_call(la, gs)
        else:
            called = False
        if called:
            if oth == 0 and not forced:
                rec["vol_calls"] += 1
            pile_before = game.get_pile_size()
            ok2, _, action = game.call_bluff(oth)
            if ok2 and action:
                if cur == 1 and oth == 0:
                    rec["wrong_cost"] += pile_before
                bots[cur].observe_action(action, game.get_hand(oth).size())
                bots[oth].observe_action(action, game.get_hand(cur).size())
        elif game.can_pass():
            okp, _ = game.pass_turn(passer=oth)
            if okp:
                pa = Action(player=oth, cards_played=[], claimed_rank=Rank.TWO,
                            was_bluff=False, bluff_called=False,
                            caller_was_right=False,
                            pile_size_before=game.get_pile_size())
                bots[cur].observe_action(pa, game.get_hand(oth).size())
                bots[oth].observe_action(pa, game.get_hand(cur).size())
        turn += 1
    rec["winner"] = game.winner if game.winner is not None else -1
    return rec


def session(bot, n, persona):
    out = []
    for _ in range(n):
        out.append(play_game(bot, persona()))
    return out


def summ(games):
    w = sum(1 for g in games if g["winner"] == 0)
    l = sum(1 for g in games if g["winner"] == 1)
    return {"w": w, "l": l, "d": len(games) - w - l,
            "vol_calls": sum(g["vol_calls"] for g in games),
            "wrong_cost": sum(g["wrong_cost"] for g in games),
            "bluffs": sum(g["beast_bluffs"] for g in games)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--persona", type=str, default="honest",
                    choices=["honest", "bayesian"])
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--out", type=str, default="data/profile_persistence.json")
    pa = ap.parse_args()
    n = pa.games
    S = pa.sessions
    persona = PERSONAS[pa.persona]
    out_path = pa.out
    random.seed(20260913)
    path = "/tmp/profile_proof.json"

    bot = AcademicBeastBot()
    sess_out = []
    rt = None
    decay = None
    for s in range(S):
        g = session(bot, n, persona)
        sess_out.append(summ(g))
        bot.mark_session_completed(lam=1.0)
        d = bot.to_dict()
        if s == 0:
            with open(path, "w") as f:
                json.dump(d, f)
            with open(path) as f:
                loaded = AcademicBeastBot.from_dict(json.load(f))
            rt = {"counters_match": (
                loaded.opp_bluffs == d["archetype_state"]["opp_bluffs"]
                and loaded.opp_honest == d["archetype_state"]["opp_honest"]
                and loaded.opp_calls == d["archetype_state"]["opp_calls"]
                and loaded.opp_passes == d["archetype_state"]["opp_passes"]),
                "sessions": loaded._sessions_observed,
                "model_actions": loaded.model.total_actions_observed}
            m0 = loaded.model.overall_bluff.mean()
            loaded.model.apply_session_decay(0.7)
            m1 = loaded.model.overall_bluff.mean()
            prior = 1.0 / 5.0
            decay = {"m0": m0, "m1": m1,
                     "ok": abs(m1 - prior) <= abs(m0 - prior)}
        bot = AcademicBeastBot.from_dict(d)

    slope = (sess_out[-1]["w"] - sess_out[0]["w"]) if len(sess_out) > 1 else 0
    out = {"sessions": sess_out, "win_slope": slope,
           "roundtrip": rt, "decay_unit": decay}
    print(json.dumps(out, indent=1))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
