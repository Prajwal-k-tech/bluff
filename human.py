"""HumanPlayer — CLI interface for playing Bluff against a bot."""

from typing import List, Tuple, Optional
from cards import Card, Hand, Rank
from game import GameState


class HumanPlayer:
    """Interactive CLI human player."""

    def __init__(self, player_id: int = 0):
        self.player_id = player_id

    def choose_play(self, hand: Hand, game: GameState) -> Tuple[List[Card], Rank]:
        """Prompt the human to select cards and claim a rank."""
        print(f"\n  Your hand ({hand.size()} cards): {hand.display()}")

        # Select cards
        while True:
            raw = input("  Select cards (comma-separated indices, e.g. 0,1,2): ").strip()
            if not raw:
                continue
            try:
                indices = [int(x.strip()) for x in raw.split(",")]
            except ValueError:
                print("  Invalid input. Use numbers separated by commas.")
                continue
            if not all(0 <= i < hand.size() for i in indices):
                print(f"  Index out of range. Valid: 0–{hand.size() - 1}")
                continue
            if len(indices) == 0:
                print("  Select at least 1 card.")
                continue
            if len(indices) > 4:
                print("  Cannot play more than 4 cards.")
                continue
            break

        cards = [hand.cards[i] for i in indices]

        # Select rank
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
            "A": Rank.ACE,
        }
        while True:
            rank_str = input("  Claim rank (2-10, J, Q, K, A): ").strip().upper()
            if rank_str in rank_map:
                break
            print("  Invalid rank. Use: 2-10, J, Q, K, A")

        return cards, rank_map[rank_str]

    def choose_call(self, hand: Hand, game: GameState,
                    opponent_played_n: int, claimed_rank: Rank,
                    pile_size: int) -> bool:
        """Ask the human whether to call bluff or pass."""
        print(f"\n  Bot played {opponent_played_n} card(s) as {claimed_rank.display()}")
        print(f"  Pile: {pile_size} cards | Your hand: {hand.size()} cards")

        can_pass = len(game.draw_pile) > 0
        if can_pass:
            options = "[C]all bluff  /  [P]ass (draw 1 card)"
        else:
            options = "[C]all bluff  (draw pile empty — cannot pass)"

        while True:
            choice = input(f"  {options}: ").strip().upper()
            if choice in ("C", "CALL"):
                return True
            if choice in ("P", "PASS") and can_pass:
                return False
            print("  Invalid choice. Enter C or P.")

    def show_result(self, message: str):
        """Display the result of a bluff call or other action."""
        print(f"\n  >> {message}")
