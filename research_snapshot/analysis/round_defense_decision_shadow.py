"""Research-only same-state decision shadow using reset defensive beliefs."""
from copy import deepcopy
from types import MethodType

from analysis.round_defense_reset_shadow import DefenseResetShadow


class DefenseDecisionShadow(DefenseResetShadow):
    def choose_action(self, hand, state):
        before = len(self.forecast_rows)
        choice = super().choose_action(hand, state)
        if len(self.forecast_rows) != before:
            row = self.forecast_rows[-1]
            probability = row['predictions']['reset_shadow']
            shadow = deepcopy(self)
            baseline, _ = shadow._plan(deepcopy(hand), deepcopy(state))
            if baseline != choice:
                raise AssertionError('same-state planner did not reproduce acting choice')
            original = shadow._posterior_bluff

            def reset_probability(this, local_hand, local_state, action):
                _, possible = original(local_hand, local_state, action)
                return probability, possible

            shadow._posterior_bluff = MethodType(reset_probability, shadow)
            alternative, _ = shadow._plan(deepcopy(hand), deepcopy(state))
            row['reset_selected'] = alternative.get('action') == 'call_bluff'
        return choice
