"""Automated bot-vs-bot test — verifies all bots work end-to-end.

Runs round-robin tournament: every bot plays every other bot.
Reports win rates and detects broken bots.
"""

import sys
import os
import random
from functools import partial
from typing import Optional
sys.path.insert(0, os.path.dirname(__file__))

from cards import Card, Hand, Rank
from game import GameState, Action
from bots.base import BotInterface, build_game_state
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from analysis.logger import GameLogger, TeeLogger

try:
    from bots.pure_nn_bot import PureNNBot
    _PURENN_AVAILABLE = True
except (ImportError, OSError):
    _PURENN_AVAILABLE = False

try:
    from bots.hybrid_bot import HybridBot
    _HYBRID_AVAILABLE = True
except (ImportError, OSError):
    _HYBRID_AVAILABLE = False

try:
    from bots.academic_beast_bot import AcademicBeastBot
    _BEAST_AVAILABLE = True
except (ImportError, OSError):
    _BEAST_AVAILABLE = False


def play_bot_vs_bot(bot_a: BotInterface, bot_b: BotInterface,
                    max_turns: int = 300,
                    logger: Optional[GameLogger] = None) -> Optional[int]:
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
                actions=game.actions,
            ),
        )

        if not cards:
            turn += 1
            continue

        success, msg = game.play_cards(current, cards, rank)
        if not success:
            turn += 1
            continue

        # Pile size at play time (a call resolves/empties the pile before we
        # log — report.py needs the at-play value for pile-size conditioning,
        # cf. Dewey et al.: bluff rates are calibrated to stakes).
        pile_at_play = game.get_pile_size()

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
                hand=game.get_hand(other).cards,
                actions=game.actions,
            ),
        )

        passed = False
        if should_call:
            success, result_msg, action = game.call_bluff(other)
            if success and action:
                # Both bots observe the revealed cards
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            # Pass: draw 1 card and advance turn
            if game.can_pass():
                success, pass_msg = game.pass_turn(passer=other)
                passed = success
                if success:
                    # Notify current bot that opponent passed (chose not to call)
                    # matching server.py:514-525 without leaking face-down cards
                    pass_act = Action(
                        player=other,
                        cards_played=[],
                        claimed_rank=Rank.TWO,
                        was_bluff=False,
                        bluff_called=False,
                        caller_was_right=False,
                        pile_size_before=game.get_pile_size(),
                    )
                    bots[current].observe_action(pass_act, game.get_hand(other).size())
            else:
                # Cannot pass when draw pile empty — force call bluff
                # (game-rules.md §2)
                success, result_msg, action = game.call_bluff(other)
                if success and action:
                    bots[current].observe_action(action, game.get_hand(other).size())
                    bots[other].observe_action(action, game.get_hand(current).size())

        if logger is not None:
            # Format-v2 semantics: exactly ONE log_action per play, logged
            # AFTER resolution so action.bluff_called/caller_was_right are
            # final. The logger emits the play record itself and, when a call
            # happened, the second call record owned by the caller. Logging
            # before resolution left bluff_called always False (v1 bug —
            # Muse methodology catch, 2026-09-10).
            logger.log_action(last_action, game, caller=other,
                              pile_size_at_play=pile_at_play)
            if passed:
                logger.log_pass(game, other)

        turn += 1

    if logger is not None:
        logger.log_game_end(game, game.winner if game.game_over else None)

    return game.winner if game.game_over else -1


def run_tournament(num_games: int = 20, log_path: Optional[str] = None,
                   checkpoint: Optional[str] = None,
                   seed: Optional[int] = None,
                   include_beast: bool = False):
    """Run round-robin tournament between all bots.

    Args:
        checkpoint: if given, PureNNBot loads this checkpoint instead of its
            default (avoids silent random-fallback when final.pt is missing).
        seed: if given, seeds Python's `random` (deck shuffles, RandomBot,
            Bayesian bluff draws, PureNN fallback). Tournament path never
            samples torch (PureNN decides deterministically), so this covers
            all stochasticity in the harness.
        include_beast: if True, includes AcademicBeastBot in the tournament.
    """
    if seed is not None:
        random.seed(seed)
    bots = {
        "Random": RandomBot,
        "Honest": HonestBot,
        "CardCount": CardCountBot,
        "Bayesian": BayesianBot,
    }
    if _PURENN_AVAILABLE:
        bots["PureNN"] = (partial(PureNNBot, checkpoint_path=checkpoint)
                           if checkpoint else PureNNBot)
    else:
        print("  [SKIP] PureNNBot not available (torch or checkpoint missing)")

    if _HYBRID_AVAILABLE:
        bots["Hybrid"] = (partial(HybridBot, checkpoint_path=checkpoint)
                           if checkpoint else HybridBot)
    else:
        print("  [SKIP] HybridBot not available")

    if include_beast and _BEAST_AVAILABLE:
        bots["AcademicBeast"] = (partial(AcademicBeastBot, checkpoint_path=checkpoint)
                                 if checkpoint else AcademicBeastBot)
    elif include_beast:
        print("  [SKIP] AcademicBeastBot not available")

    tee = None
    if log_path:
        tee = TeeLogger(log_path)
        print(f"  Logging all actions to {log_path}")

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

            for g in range(num_games):
                # 50/50 seat alternation to eliminate positional bias
                if g % 2 == 0:
                    bot_0, bot_1 = bots[name_a](), bots[name_b]()
                    p0_name, p1_name = name_a, name_b
                else:
                    bot_0, bot_1 = bots[name_b](), bots[name_a]()
                    p0_name, p1_name = name_b, name_a

                logger = tee.new_game(p0_name, p1_name) if tee else None
                result = play_bot_vs_bot(bot_0, bot_1, logger=logger)

                if result == 0:
                    winner_name, loser_name = p0_name, p1_name
                elif result == 1:
                    winner_name, loser_name = p1_name, p0_name
                else:
                    winner_name, loser_name = None, None

                if winner_name == name_a:
                    a_wins += 1
                    wins[name_a] += 1
                    losses[name_b] += 1
                elif winner_name == name_b:
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
    parser.add_argument("--log", type=str, default=None,
                        help="JSONL path to log all actions "
                             "(e.g. data/terminal/tournament.jsonl)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint for PureNNBot "
                             "(e.g. nn/checkpoints/v4.pt). Without it, "
                             "PureNNBot uses its default final.pt and falls "
                             "back to random play if missing.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed Python RNG for reproducible tournaments "
                             "(deck shuffles + rule-bot draws).")
    parser.add_argument("--include-beast", action="store_true",
                        help="Include AcademicBeastBot (Dewey EV + Southey) in the tournament.")
    args = parser.parse_args()
    run_tournament(args.games, log_path=args.log,
                   checkpoint=args.checkpoint, seed=args.seed,
                   include_beast=args.include_beast)


if __name__ == "__main__":
    main()
