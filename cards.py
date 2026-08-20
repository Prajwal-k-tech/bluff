"""Card, Deck, and Hand for 2-player Bluff."""

import random
from enum import IntEnum
from typing import List, Optional


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def display(self) -> str:
        names = {
            2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
            8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"
        }
        return names[self.value]


class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

    def display(self) -> str:
        return ["♥", "♦", "♣", "♠"][self.value]


class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self) -> str:
        return f"{self.rank.display()}{self.suit.display()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def __lt__(self, other) -> bool:
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit


class Deck:
    def __init__(self):
        self.cards: List[Card] = []
        self.reset()

    def reset(self):
        self.cards = [Card(rank, suit) for suit in Suit for rank in Rank]
        random.shuffle(self.cards)

    def deal(self, n: int) -> List[Card]:
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def remaining(self) -> int:
        return len(self.cards)


class Hand:
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards: List[Card] = sorted(cards) if cards else []

    def add(self, cards: List[Card]):
        self.cards.extend(cards)
        self.cards.sort()

    def remove(self, cards: List[Card]) -> bool:
        """Remove specific cards from hand. Returns False if any card not found."""
        remaining = list(self.cards)
        for c in cards:
            if c in remaining:
                remaining.remove(c)
            else:
                return False
        self.cards = remaining
        return True

    def size(self) -> int:
        return len(self.cards)

    def has_cards(self) -> bool:
        return len(self.cards) > 0

    def display(self) -> str:
        if not self.cards:
            return "(empty)"
        return " ".join(str(c) for c in self.cards)

    def ranks_available(self) -> dict:
        """Count how many cards of each rank we hold."""
        counts = {}
        for c in self.cards:
            counts[c.rank] = counts.get(c.rank, 0) + 1
        return counts
