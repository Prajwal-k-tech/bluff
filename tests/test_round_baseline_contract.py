"""Small deterministic contract checks for the corrected baseline policies."""

from math import comb

from bots.round_policy import RoundHonestBot, RoundMathBot, RoundRandomBot
from cards import Card, Rank, Suit
from game import Action


def c(rank, suit=Suit.HEARTS):
    return Card(rank, suit)


def state(*, active=None, last=None, pile=0, opponent=5,
          can_play=True, can_call=False, can_pass=False):
    return {
        "viewer": 0,
        "player_id": 0,
        "active_rank": active,
        "opponent_hand_size": opponent,
        "pile_size": pile,
        "draw_pile_size": int(can_pass),
        "actions": [last] if last is not None else [],
        "last_action": last,
        "own_pile_cards": [],
        "can_play": can_play,
        "can_call_bluff": can_call,
        "can_pass": can_pass,
    }


def test_random_uses_hierarchical_family_rank_quantity_subset_draws():
    class ScriptedRandom:
        def __init__(self):
            self.calls = []

        def choice(self, values):
            self.calls.append(("choice", values))
            if values and values[0] == "play":
                return "play"
            return Rank.QUEEN

        def randint(self, low, high):
            self.calls.append(("randint", low, high))
            return 2

        def sample(self, population, quantity):
            self.calls.append(("sample", population, quantity))
            return [population[0], population[2]]

    hand = [c(Rank.TWO), c(Rank.FOUR), c(Rank.SEVEN)]
    bot = RoundRandomBot()
    bot.rng = ScriptedRandom()

    result = bot.choose_action(hand, state())

    assert result == {"action": "play", "cards": [hand[0], hand[2]],
                      "rank": Rank.QUEEN}
    assert bot.rng.calls == [
        ("choice", ["play"]),
        ("choice", tuple(Rank)),
        ("randint", 1, 3),
        ("sample", hand, 2),
    ]

    # Midround, the rank draw disappears and the active rank is mandatory.
    bot.rng = ScriptedRandom()
    result = bot.choose_action(hand, state(active=Rank.NINE))
    assert result["rank"] == Rank.NINE
    assert [call[0] for call in bot.rng.calls] == ["choice", "randint", "sample"]


def test_honest_only_lies_in_a_custom_no_pass_no_truth_position():
    # An empty supply is unreachable under ordinary recycling rules. This is a
    # defensive custom state: no truthful active-rank card, no pass, and no
    # proven contradiction to justify a challenge. Playing therefore requires
    # a lie under the policy's no-unproven-calls contract.
    last = Action(1, [None], Rank.SEVEN, None, False, False, 0)
    hand = [c(Rank.TWO), c(Rank.FOUR)]
    custom = state(active=Rank.SEVEN, last=last, opponent=3,
                   can_play=True, can_call=True, can_pass=False)

    bot = RoundHonestBot()
    result = bot.choose_action(hand, custom)

    assert result == {"action": "play", "cards": [hand[0]], "rank": Rank.SEVEN}
    assert bot.last_decision_telemetry["reason"] == "forced_only_play"


def test_math_bluff_posterior_matches_direct_combination_oracle():
    # Our hand and the current public packet leave 50 unknown cards, 3 of them
    # Sevens. Before its two-card packet, the opponent had 5 cards. Count all
    # five-card subsets directly rather than calling the policy's tail helper.
    hand = [c(Rank.SEVEN), c(Rank.TWO)]
    last = Action(1, [None, None], Rank.SEVEN, None, False, False, 0)
    last.action_index = 0
    view = state(active=Rank.SEVEN, last=last, pile=2, opponent=3,
                 can_play=False, can_call=True, can_pass=True)

    # P(at least two Sevens) = (C(3,2)C(47,3) + C(3,3)C(47,2))/C(50,5).
    can_truthfully_hold = (
        comb(3, 2) * comb(47, 3) + comb(3, 3) * comb(47, 2)
    ) / comb(50, 5)
    expected_bluff_posterior = 0.30 / (
        0.30 + 0.70 * can_truthfully_hold
    )

    posterior, modeled_truth_support = RoundMathBot()._posterior_bluff(
        hand, view, last
    )

    assert modeled_truth_support == can_truthfully_hold
    assert posterior == expected_bluff_posterior


def test_math_scores_play_vs_pass_and_prefers_truthful_action_on_exact_tie():
    class ControlledMath(RoundMathBot):
        def _call_rate(self, quantity, pile_size):
            return 0.10

        def _certifiable_probability(self, hand, state, rank, quantity):
            return 0.25

    # No truthful Seven is held, so all plays are bluffs. p_call is max(.10,.25).
    # For q=2: 2 - .25*(4+2) = .5. q=1 scores -.25, q=3 is final and scores
    # 3-(4+3) = -4, and pass scores -1. Thus the two-card bluff must win.
    hand = [c(Rank.TWO), c(Rank.FOUR), c(Rank.KING)]
    planner = ControlledMath()
    selected = planner.choose_action(
        hand,
        state(active=Rank.SEVEN, pile=4, opponent=3,
              can_play=True, can_pass=True),
    )
    assert selected["action"] == "play"
    assert len(selected["cards"]) == 2
    assert planner.last_decision_telemetry["call_probability"] == 0.25
    assert planner.last_decision_telemetry["utility"] == 0.5

    # With a two-card hand containing one truthful Seven and one off-rank card,
    # q=1 truthful and bluff scores tie at 1 when modeled call risk is zero.
    # q=2 would be a caught final bluff (utility 0); pass is -1. Truth wins the
    # exact score tie, independently of physical candidate order.
    class NoRiskMath(RoundMathBot):
        def _call_rate(self, quantity, pile_size):
            return 0.0

        def _certifiable_probability(self, hand, state, rank, quantity):
            return 0.0

    one_truth = [c(Rank.SEVEN), c(Rank.TWO)]
    tie_planner = NoRiskMath()
    tied = tie_planner.choose_action(
        one_truth,
        state(active=Rank.SEVEN, pile=0, opponent=2,
              can_play=True, can_pass=True),
    )
    assert tied == {"action": "play", "cards": [one_truth[0]],
                    "rank": Rank.SEVEN}
    assert tie_planner.last_decision_telemetry["truthful"] is True
    assert tie_planner.last_decision_telemetry["utility"] == 1.0
