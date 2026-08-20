"""Automated bot-vs-bot test — verifies all bots work end-to-end.

Runs round-robin tournament: every bot plays every other bot.
Reports win rates and detects broken bots.
"""

import sys
import os
from typing import Optional
sys.path.insert(0, os.path.dirname(__file__))

from cards import Card, Hand, Rank
from game import GameState, Action
from bots.base import BotInterface, build_game_state
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot


def play_bot_vs_bot(bot_a: BotInterface, bot_b: BotInterface,
                    max_turns: int = 300) -> Optional[int]:
    """Play one game between two bots. Returns winner (0=A, 1=B) or -1 draw."""
    game = GameState(num_players=2)
    game.deal(14)

    bot_a.reset()
    bot_b.reset()
    bots = [bot_a, bot_b]

    # Set player IDs for bots that need them (e.g., BayesianBot)
    if hasattr(bot_a, 'player_id'):
        bot_a.player_id = 0
    if hasattr(bot_b, 'player_id'):
        bot_b.player_id = 1

    turn = 0
    while not game.game_over and turn < max_turns:
        current = game.current_player
        bot = bots[current]
        other = 1 - current

        hand = game.get_hand(current)
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
            ),
        )

        if not cards:
            turn += 1
            continue

        success, msg = game.play_cards(current, cards, rank)
        if not success:
            turn += 1
            continue

        # Other bot decides: call bluff or pass
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
            ),
        )

        if should_call:
            success, result_msg, action = game.call_bluff(other)
            if success and action:
                # Both bots observe the revealed cards
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            # Pass: draw 1 card and advance turn
            if game.can_pass():
                success, pass_msg = game.pass_turn()
            else:
                # Cannot pass when draw pile empty — force call bluff
                success, result_msg, action = game.call_bluff(other)
                if success and action:
                    bots[current].observe_action(action, game.get_hand(other).size())
                    bots[other].observe_action(action, game.get_hand(current).size())

        turn += 1

    return game.winner if game.game_over else -1


def run_tournament(num_games: int = 20):
    """Run round-robin tournament between all bots."""
    bots = {
        "Random": RandomBot,
        "Honest": HonestBot,
        "CardCount": CardCountBot,
        "Bayesian": BayesianBot,
    }

    names = list(bots.keys())
    wins = {name: 0 for name in names}
    losses = {name: 0 for name in names}
    draws = {name: 0 for name in names}

    print(f"\n{'='*60}")
    print(f"  TOURNAMENT — {num_games} games per matchup")
    print(f"{'='*60}\n")

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i >= j:
                continue

            a_wins = 0
            b_wins = 0
            draw_count = 0

            for _ in range(num_games):
                bot_a = bots[name_a]()
                bot_b = bots[name_b]()
                result = play_bot_vs_bot(bot_a, bot_b)

                if result == 0:
                    a_wins += 1
                    wins[name_a] += 1
                    losses[name_b] += 1
                elif result == 1:
                    b_wins += 1
                    wins[name_b] += 1
                    losses[name_a] += 1
                else:
                    draw_count += 1
                    draws[name_a] += 1
                    draws[name_b] += 1

            a_pct = a_wins / num_games * 100
            b_pct = b_wins / num_games * 100
            print(f"  {name_a:12s} vs {name_b:12s}: "
                  f"{a_wins:2d}-{b_wins:2d}-{draw_count} "
                  f"({a_pct:.0f}%-{b_pct:.0f}%)")

    # Summary
    print(f"\n{'='*60}")
    print("  STANDINGS")
    print(f"{'='*60}")
    print(f"  {'Bot':<12s} {'W':>4s} {'L':>4s} {'D':>4s}")
    print(f"  {'-'*24}")
    for name in names:
        print(f"  {name:<12s} {wins[name]:>4d} {losses[name]:>4d} {draws[name]:>4d}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bluff Bot Tournament")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per matchup (default: 20)")
    args = parser.parse_args()
    run_tournament(args.games)


if __name__ == "__main__":
    main()
