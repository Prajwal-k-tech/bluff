from analysis.round_arena import play_game
from bots.round_current_deal_defense_bot import RoundCurrentDealDefenseBot
from bots.round_opening_public_expert_bot import RoundPrunedOpeningPublicExpertBot
from bots.round_policy import RoundMathBot


def test_first_game_matches_baseline_and_profile_roundtrips():
    runs = []
    for cls in (RoundPrunedOpeningPublicExpertBot, RoundCurrentDealDefenseBot):
        bot, opponent = cls(11), RoundMathBot(12)
        result = play_game([bot, opponent], seed=2416001001, starting_player=0, max_decisions=256)
        runs.append((result.winner, result.decisions, bot.rng.getstate(), opponent.to_dict()))
        clone = cls(99)
        clone.restore_opponent_profile(bot.to_dict())
        assert clone.to_dict() == bot.to_dict()
    assert runs[0] == runs[1]
