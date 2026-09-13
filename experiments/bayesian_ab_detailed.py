"""A/B validation with forced-call tracking."""
import sys, os, random, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState
from bots.base import build_game_state
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot


def play_instrumented(bot_a, bot_b, seed):
    random.seed(seed)
    game = GameState(num_players=2)
    game.deal(14)
    bots = [bot_a, bot_b]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()
    game.current_player = seed % 2

    s = [{"shed": 0, "plays": 0, "calls": 0, "wrong": 0, "took": 0, "resp": 0,
          "forced": 0, "vol_calls": 0}
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
            can_pass = game.can_pass()
            should_call = (not can_pass) or bots[other].decide_call(last_action, gs)
            is_forced = should_call and not can_pass
        else:
            should_call = False
            is_forced = False
        if should_call:
            s[other]["calls"] += 1
            if is_forced:
                s[other]["forced"] += 1
            else:
                s[other]["vol_calls"] += 1
            before = game.get_hand(other).size()
            ok2, _, action = game.call_bluff(other)
            if ok2 and action:
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


def run_matchup(name_a, make_a, name_b, make_b, games=100, seed=7):
    agg_w = agg_l = agg_d = 0
    agg = {"A": {"shed": 0, "plays": 0, "calls": 0, "wrong": 0, "took": 0, "resp": 0,
                 "forced": 0, "vol_calls": 0},
           "B": {"shed": 0, "plays": 0, "calls": 0, "wrong": 0, "took": 0, "resp": 0,
                 "forced": 0, "vol_calls": 0}}
    turns = []
    for i in range(games):
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

    a = agg["A"]
    call_r = a["calls"] / max(1, a["resp"])
    wrong_r = a["wrong"] / max(1, a["calls"])
    took_g = a["took"] / games
    vol = a["vol_calls"]
    forced = a["forced"]
    print(f"  {name_a} vs {name_b}: {agg_w}-{agg_l}-{agg_d} | "
          f"call {call_r:.1%} (vol={vol} forced={forced}) | wrong {wrong_r:.1%} | "
          f"took/game {took_g:.1f} | avg turns {statistics.mean(turns):.0f}")


if __name__ == "__main__":
    N = 100
    print(f"=== BayesianBot A/B Test (N={N}, seats swapped) ===\n")
    matchups = [
        ("Bayesian", lambda: BayesianBot(), "Honest", lambda: HonestBot()),
        ("Bayesian", lambda: BayesianBot(), "Random", lambda: RandomBot()),
        ("Bayesian", lambda: BayesianBot(), "CardCount", lambda: CardCountBot()),
        ("Bayesian", lambda: BayesianBot(), "Bayesian", lambda: BayesianBot()),
    ]
    for name_a, make_a, name_b, make_b in matchups:
        run_matchup(name_a, make_a, name_b, make_b, games=N)
