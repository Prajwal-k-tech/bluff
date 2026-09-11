"""
Claim-pattern early-discriminator study — can observable claim statistics
separate maniac-class opponents from honest/balanced ones BEFORE the bluff
posterior forms (which never crosses 0.35 in-game)?

Motivation (claims doc §ADR-012): the conditioned gate's activation is
starved — the posterior mean crawls to ~0.30 vs a 0.75-bluff maniac by turn
95. But claim patterns are PUBLIC from turn 0: packet sizes, claims per
turn, rank-repetition. If a claim-pattern statistic separates the persona
classes early, the conditioned gate gains its missing early signal.

Method: real games vs the 4 decisive personas; per-turn logging of the
opponent's claim sizes; aggregate the first-K-opponent-plays statistics;
compare distributions across personas.

Run:  python3 experiments/claim_pattern_discriminator.py [games_per_persona]
"""
import sys, os, statistics, random
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


def play_and_trace(make_opponent, measured_seat, max_turns=100):
    """One game; returns the opponent's per-play claim sizes (in order)."""
    game = GameState(num_players=2)
    game.deal(14)
    measured = HybridBot(thompson_sampling=True)
    opponent = make_opponent()
    bots = [measured, opponent]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        for b2 in bots:
            if hasattr(b2, "reset"):
                b2.reset()

    turn = 0
    claim_sizes = []  # opponent's claim sizes, in order
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
        success, _ = game.play_cards(current, cards, rank)
        if not success:
            turn += 1
            continue

        last_action = game.actions[-1]
        if last_action.player == (1 - measured_seat):
            claim_sizes.append(len(last_action.cards_played))

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

    return claim_sizes


def main(n_games: int = 30):
    personas = ["Passive_Honest", "Balanced_Standard", "Hyper_Maniac", "Total_Maniac_Extreme"]
    print(f"Claim-pattern discriminator study — {n_games} games/persona "
          f"(opponent claim sizes, in-game order)\n")

    # Per-persona: the first-8-plays mean claim size distribution
    stats = {}
    for pname in personas:
        first8_means, overall_sizes = [], []
        for g in range(n_games):
            sizes = play_and_trace(make_persona(pname), g % 2)
            if len(sizes) >= 4:
                first8_means.append(statistics.mean(sizes[:8]))
            overall_sizes.extend(sizes)
        stats[pname] = {"first8": first8_means, "sizes": overall_sizes}
        print(f"{pname:>22}: mean claim size (first 8 plays) "
              f"{statistics.mean(first8_means):.2f} "
              f"[{min(first8_means):.2f}-{max(first8_means):.2f}] | "
              f"overall mean {statistics.mean(overall_sizes):.2f} "
              f"| plays/game {len(overall_sizes)/n_games:.1f}")

    print("\nDiscriminator read: if the maniac personas' first-8 means sit "
          "clearly above the honest/balanced ones, a packet-size gate is "
          "viable. Overlap means claim patterns do NOT separate early — "
          "the honest answer stays 'activation latency is structural'.")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n)
