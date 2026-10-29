"""Research-only actor using current-deal evidence for defensive beliefs."""
from bots.round_opening_public_expert_bot import RoundPrunedOpeningPublicExpertBot


class RoundCurrentDealDefenseBot(RoundPrunedOpeningPublicExpertBot):
    """Keep offense profile; reconstruct defense from this deal's legal view."""
    policy_name = 'round_current_deal_defense_research_v1'

    def _posterior_bluff(self, hand, state, action):
        shadow = RoundPrunedOpeningPublicExpertBot(0)
        shadow.player_id = self.player_id
        shadow.observe_player_view(state)
        return shadow._posterior_bluff(hand, state, action)
