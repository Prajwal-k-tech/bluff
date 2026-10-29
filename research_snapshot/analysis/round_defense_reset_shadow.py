"""Same-view reset-profile defensive prediction, synthetic research only."""
from analysis.round_predecision_bluff_audit import traced_policy
from bots.round_opening_public_expert_bot import RoundPrunedOpeningPublicExpertBot


class DefenseResetShadow(traced_policy(RoundPrunedOpeningPublicExpertBot)):
    """Reconstruct a fresh-profile shadow from each legal current-deal view."""
    def _posterior_bluff(self, hand, state, action):
        result = super()._posterior_bluff(hand, state, action)
        shadow = RoundPrunedOpeningPublicExpertBot(0)
        shadow.player_id = self.player_id
        shadow.observe_player_view(state)
        self._reset_prediction = shadow._posterior_bluff(hand, state, action)[0]
        return result

    def choose_action(self, hand, state):
        self._reset_prediction = None
        count = len(self.forecast_rows)
        choice = super().choose_action(hand, state)
        if len(self.forecast_rows) != count:
            if self._reset_prediction is None:
                raise AssertionError('reset shadow forecast missing')
            self.forecast_rows[-1]['predictions']['reset_shadow'] = self._reset_prediction
        return choice
