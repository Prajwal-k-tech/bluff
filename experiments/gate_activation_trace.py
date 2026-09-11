"""
Gate-activation latency tracer — finishes the Total_Maniac activation-latency
todo (claims doc §ADR-012 action item; diagnosed axis: posterior-formation
latency, floor constant ruled out by matrix v4).

Plays conditioned HybridBot vs Total_Maniac_Extreme and records, at every
respond decision, the model's inferred bluff-rate posterior mean and the
gate state. Reports: activation turn distribution, pre-activation share of
games, and the posterior formation curve.

Run:  python3 experiments/gate_activation_trace.py [n_games]
"""
import sys, os, random, statistics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import GameState, Action
from cards import Rank
from bots.base import build_game_state
from bots.hybrid_bot import HybridBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot


def make_persona(name):
    proto = next(p for p in SYNTHETIC_POPULATION if p.name == name)
    return lambda: PersonaBot(proto.name, proto.true_bluff_rate,
                              proto.true_call_rate, proto.bluff_size_pref,
                              proto.honest_dump_multi)


def trace_game(make_measured, make_opponent, measured_seat, max_turns=100):
    game = GameState(num_players=2)
    game.deal(14)
    bots = [make_measured(), make_opponent()]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()

    turn = 0
    trace = []          # (engine_turn, posterior_mean, gate_on)
    activation = None   # first engine turn with gate ON
    while not game.game_over and turn < max_turns:
        current = game.current_player
        other = 1 - current
        if current == measured_seat:
            ob = getattr(bots[measured_seat].model, "overall_bluff", None)
            if ob is not None:
                gate = bots[measured_seat]._use_thompson()
                trace.append((game.turn_count, ob.mean(), gate))
                if gate and activation is None:
                    activation = game.turn_count

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
        success, _ = game.play_cards(current, cards, rank)
        if not success:
            turn += 1
            continue

        last_action = game.actions[-1]
        should_call = bots[other].decide_call(
            last_action=last_action,
            game_state=build_game_state(
                hand_size=game.get_hand(other).size(),
                opponent_hand_size=game.get_hand(current).size(),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=turn,
                last_action=last_action,
                cards_played=game.get_cards_played(),
                hand=game.get_hand(other).cards,
                actions=game.actions,
            ),
        )
        if should_call:
            success, _, action = game.call_bluff(other)
            if success and action:
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            if game.can_pass():
                game.pass_turn(passer=other)
                pass_act = Action(player=other, cards_played=[], claimed_rank=Rank.TWO,
                                  was_bluff=False, bluff_called=False,
                                  caller_was_right=False,
                                  pile_size_before=game.get_pile_size())
                bots[current].observe_action(pass_act, game.get_hand(other).size())
                bots[other].observe_action(pass_act, game.get_hand(current).size())
        turn += 1

    return {"activation": activation, "trace": trace,
            "winner": game.winner, "turns": game.turn_count}


def main(n_games: int = 20, seed: int = 20260914):
    random.seed(seed)
    make_maniac = make_persona("Total_Maniac_Extreme")

    activations, never = [], 0
    curve = {}
    wins = 0
    for g in range(n_games):
        seat = g % 2
        if seat == 0:
            r = trace_game(lambda: HybridBot(thompson_sampling=True), make_maniac, 0)
        else:
            r = trace_game(make_maniac, lambda: HybridBot(thompson_sampling=True), 1)
        if r["activation"] is not None:
            activations.append(r["activation"])
        else:
            never += 1
        if r["winner"] == seat:
            wins += 1
        for t, mean, gate in r["trace"]:
            curve.setdefault(t, []).append(mean)

    print(f"Gate-activation trace — {n_games} games vs Total_Maniac_Extreme "
          f"(conditioned HybridBot, seed {seed})")
    if activations:
        print(f"  activated: {len(activations)}/{n_games} games "
              f"({100*len(activations)/n_games:.0f}%)")
        print(f"  activation turn: median {statistics.median(activations):.0f}, "
              f"mean {statistics.mean(activations):.1f}, "
              f"range {min(activations)}-{max(activations)}")
    else:
        print(f"  activated: 0/{n_games} games")
    print(f"  never activated: {never} games")
    print(f"  conditioned wins: {wins}/{n_games}")
    print("\n  posterior-mean formation curve (turn: mean ± spread):")
    for t in sorted(curve):
        vals = curve[t]
        if len(vals) >= 3 and t % 5 == 0:
            print(f"    turn {t:>3}: {statistics.mean(vals):.2f} "
                  f"({min(vals):.2f}-{max(vals):.2f}) n={len(vals)}")
    total_turns = sum(len(v) for v in curve.values())
    pre = sum(1 for t, vals in curve.items() for _ in vals if t < (min(activations) if activations else 10**9))
    if activations:
        print(f"\n  pre-activation decisions: {pre}/{total_turns} "
              f"({100*pre/total_turns:.0f}% of the trace)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
