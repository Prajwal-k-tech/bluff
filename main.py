"""Main game runner — ties everything together.

Usage:
    python main.py                          # Interactive game vs BayesianBot
    python main.py --bot random             # Play against RandomBot
    python main.py --bot honest             # Play against HonestBot
    python main.py --bot cardcount          # Play against CardCountBot
    python main.py --bot bayesian           # Play against BayesianBot (default)
    python main.py --games 20               # Play multiple games
    python main.py --stats                  # Show model stats after playing
"""

import os
import sys
import argparse
from typing import Optional

from cards import Card, Hand, Rank
from game import GameState, Action
from human import HumanPlayer
from bots.base import BotInterface, build_game_state

# Bot registry
BOT_REGISTRY = {
    "random": ("bots.random_bot", "RandomBot"),
    "honest": ("bots.honest_bot", "HonestBot"),
    "cardcount": ("bots.cardcount_bot", "CardCountBot"),
    "bayesian": ("bots.bayesian_bot", "BayesianBot"),
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def create_bot(name: str) -> BotInterface:
    """Create a bot by name from the registry."""
    if name not in BOT_REGISTRY:
        raise ValueError(f"Unknown bot: {name}. Choose from: {list(BOT_REGISTRY.keys())}")
    module_path, class_name = BOT_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    bot = cls()
    # Load saved state if bayesian
    if name == "bayesian":
        model_path = os.path.join(MODEL_DIR, f"bayesian_{class_name.lower()}.json")
        if os.path.exists(model_path):
            bot.load(model_path)
    return bot


def play_game(human_player: HumanPlayer, bot: BotInterface,
              human_id: int = 0, verbose: bool = True) -> Optional[int]:
    """Play one full game of 2-player Bluff. Returns winner or None if draw."""
    game = GameState(num_players=2)
    game.deal(14)  # 14 cards each, 24 in draw pile
    bot.reset()

    bot_id = 1 - human_id

    if verbose:
        print("\n" + "=" * 55)
        print("  NEW GAME — First to empty hand wins!")
        print("=" * 55)
        print(f"  Your hand ({game.get_hand(human_id).size()} cards): "
              f"{game.get_hand(human_id).display()}")
        print(f"  Bot has {game.get_hand(bot_id).size()} cards.")

    turn = 0
    max_turns = 200

    while not game.game_over and turn < max_turns:
        current = game.current_player
        turn += 1

        if verbose:
            game.display_state(human_id)

        if current == human_id:
            # Human plays
            hand = game.get_hand(human_id)
            cards, rank = human_player.choose_play(hand, game)

            success, msg = game.play_cards(human_id, cards, rank)
            if not success:
                print(f"  Error: {msg}")
                continue

            # Bot decides: call bluff or pass
            should_call = bot.decide_call(
                last_action=game.actions[-1],
                game_state=build_game_state(
                    hand_size=game.get_hand(bot_id).size(),
                    opponent_hand_size=game.get_hand(human_id).size(),
                    pile_size=game.get_pile_size(),
                    draw_pile_size=len(game.draw_pile),
                    turn_number=turn,
                    last_action=game.actions[-1],
                    cards_played=game.get_cards_played(),
                ),
            )

            if should_call:
                success, result_msg, action = game.call_bluff(bot_id)
                if success:
                    human_player.show_result(result_msg)
                    if action:
                        bot.observe_action(action, game.get_hand(human_id).size())
            else:
                if verbose:
                    print("  Bot passes.")

        else:
            # Bot plays
            bot_hand = game.get_hand(bot_id)
            cards, rank = bot.decide_play(
                hand=bot_hand.cards,
                game_state=build_game_state(
                    hand_size=bot_hand.size(),
                    opponent_hand_size=game.get_hand(human_id).size(),
                    pile_size=game.get_pile_size(),
                    draw_pile_size=len(game.draw_pile),
                    turn_number=turn,
                    last_action=game.actions[-1] if game.actions else None,
                    cards_played=game.get_cards_played(),
                ),
            )

            if not cards:
                continue

            success, msg = game.play_cards(bot_id, cards, rank)
            if not success:
                if verbose:
                    print(f"  Bot error: {msg}")
                continue

            if verbose:
                print(f"\n  Bot plays {len(cards)} card(s) as {rank.display()}")

            # Human decides: call bluff or pass
            human_choice = human_player.choose_call(
                hand=game.get_hand(human_id),
                game=game,
                opponent_played_n=len(cards),
                claimed_rank=rank,
                pile_size=game.get_pile_size(),
            )

            if human_choice:
                success, result_msg, action = game.call_bluff(human_id)
                if success:
                    human_player.show_result(result_msg)
                    if action:
                        bot.observe_action(action, game.get_hand(bot_id).size())
            else:
                # Human passed — advance turn, bot observes the play
                game.pass_turn()
                was_bluff = not all(c.rank == rank for c in cards)
                action = Action(
                    player=bot_id,
                    cards_played=list(cards),
                    claimed_rank=rank,
                    was_bluff=was_bluff,
                    bluff_called=False,
                    caller_was_right=False,
                    pile_size_before=game.get_pile_size() - len(cards),
                )
                bot.observe_action(action, game.get_hand(human_id).size())

    if game.game_over:
        if verbose:
            winner_name = "You" if game.winner == human_id else "Bot"
            print(f"\n{'='*55}")
            print(f"  GAME OVER — {winner_name} wins!")
            print(f"{'='*55}")
        return game.winner
    else:
        if verbose:
            print("\n  Game ended in a draw (too many turns).")
        return None


def run_session(bot_name: str, num_games: int = 1):
    """Run a session of multiple games."""
    print("\n" + "=" * 55)
    print("  BLUFF — The Card Game of Deception")
    print(f"  Opponent: {bot_name} bot")
    print("=" * 55)

    bot = create_bot(bot_name)
    human = HumanPlayer(player_id=0)

    stats = {"wins": 0, "losses": 0, "draws": 0}

    for game_num in range(1, num_games + 1):
        if num_games > 1:
            print(f"\n--- Game {game_num}/{num_games} ---")

        winner = play_game(human, bot, human_id=0, verbose=True)

        if winner == 0:
            stats["wins"] += 1
        elif winner == 1:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        # Save bayesian bot state
        if bot_name == "bayesian":
            os.makedirs(MODEL_DIR, exist_ok=True)
            bot.save(os.path.join(MODEL_DIR, "bayesian_bayesianbot.json"))

        if num_games > 1 and game_num < num_games:
            cont = input("\nPress Enter for next game, or 'q' to quit: ").strip()
            if cont.lower() == "q":
                break

    # Final stats
    print(f"\n{'='*55}")
    print("  SESSION RESULTS")
    print("=" * 55)
    total = stats["wins"] + stats["losses"] + stats["draws"]
    print(f"  Games played: {total}")
    print(f"  Your wins: {stats['wins']}")
    print(f"  Bot wins: {stats['losses']}")
    print(f"  Draws: {stats['draws']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Bluff — Card Game with AI Bot")
    parser.add_argument("--bot", choices=list(BOT_REGISTRY.keys()), default="bayesian",
                        help="Bot to play against (default: bayesian)")
    parser.add_argument("--games", type=int, default=1,
                        help="Number of games to play (default: 1)")
    parser.add_argument("--list-bots", action="store_true",
                        help="List available bots and exit")
    args = parser.parse_args()

    if args.list_bots:
        print("Available bots:")
        for name in BOT_REGISTRY:
            print(f"  {name}")
        return

    run_session(args.bot, args.games)


if __name__ == "__main__":
    main()
