"""Research baseline-compatible adaptive bot using collector-private labels.

The frequency estimator remains heuristic and observation-selected. Extra
factual labels do not make its rates an unbiased population bluff estimate.
"""
from bots.round_policy import RoundAdaptiveBot, _context
from bots.round_private_pickup_evidence import extract_private_pickup_evidence


class RoundPickupAdaptiveBot(RoundAdaptiveBot):
    policy_name = "round_adaptive_private_pickup_v1"

    def observe_player_view(self, state):
        self._set_viewer(state)
        # Validate all collector authorization before committing any counts.
        evidence = extract_private_pickup_evidence(state)
        super().observe_player_view(state)
        for row in evidence:
            index = row["action_index"]
            action = state["actions"][index]
            key = ("action", index)
            if (key in self._resolved_action_objects
                    or action.response_was_forced is not False):
                continue
            context = _context(row["quantity"], action.pile_size_before+row["quantity"])
            self._add(self._bluff_evidence, context, row["was_bluff"])
            self._resolved_action_objects.add(key)

    def choose_action(self, hand, state):
        self.observe_player_view(state)
        return self._choose_action_after_observe(hand,state)

    def _choose_action_after_observe(self,hand,state):
        """Simulator-only: exact state was just delivered through observe."""
        return super().choose_action(hand, state)
