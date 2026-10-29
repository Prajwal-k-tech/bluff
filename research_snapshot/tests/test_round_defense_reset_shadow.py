from analysis.round_defense_reset_shadow import DefenseResetShadow
from analysis.round_defense_decision_shadow import DefenseDecisionShadow
from analysis.round_predecision_bluff_audit import _game_trace, traced_policy
from bots.round_opening_public_expert_bot import RoundPrunedOpeningPublicExpertBot
from bots.round_policy import RoundMathBot
from bots.round_policy import RoundRandomBot
import pytest


@pytest.mark.parametrize('seat,opponent_type,seed', [
    (0, RoundMathBot, 2407001001), (1, RoundMathBot, 2407001002),
    (0, RoundRandomBot, 2407001003), (1, RoundRandomBot, 2407001004),
])
def test_reset_shadow_scores_same_claims_without_changing_game(seat, opponent_type, seed):
    traces = []
    for cls in (traced_policy(RoundPrunedOpeningPublicExpertBot), DefenseResetShadow):
        actor, opponent = cls(9), opponent_type(10)
        trace = _game_trace(actor, opponent, seat=seat, deal_seed=seed,
                            recycle_seed=seed+200, starting_player=0, max_decisions=256)
        traces.append((trace['winner'], trace['decisions'], actor.to_dict(),
                       opponent.to_dict(), actor.rng.getstate()))
        if cls is DefenseResetShadow:
            assert trace['rows']
            assert all(0 <= row['predictions']['reset_shadow'] <= 1 for row in trace['rows'])
            assert all(row['predictions']['reset_shadow'] == row['predictions']['posterior']
                       for row in trace['rows'])
    assert traces[0] == traces[1]


def test_decision_shadow_matches_fresh_profile_decisions():
    actor = DefenseDecisionShadow(9)
    trace = _game_trace(actor, RoundMathBot(10), seat=0, deal_seed=2407001001,
                        recycle_seed=2407001201, starting_player=0, max_decisions=256)
    assert trace['rows']
    assert all(row['selected'] == row['reset_selected'] for row in trace['rows'])
