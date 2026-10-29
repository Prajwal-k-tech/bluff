"""Center opponent with an externally scheduled voluntary call-risk floor."""
from copy import copy, deepcopy

from bots.round_center_policy_bot import RoundCenterPolicyBot


class RoundCompetitiveSwitchActor(RoundCenterPolicyBot):
    policy_name = "round_competitive_switch_actor_v1"
    SWITCH_CONFIG = {"policy_id": policy_name, "floor": 0.8,
                     "first_games": [0, 1, 2], "switched_games": [3, 4, 5]}

    def __init__(self, seed=None, game_index=0):
        self._validate_index(game_index)
        self.game_index = game_index
        self._switch_active = game_index % 6 >= 3
        self._switch_call_rate_calls = 0
        self._switch_floor_raises = 0
        self._switch_actual_choice_changes = 0
        super().__init__(seed)

    @staticmethod
    def _validate_index(value):
        if type(value) is not int or value < 0:
            raise ValueError("game_index must be a nonnegative int")

    def set_game_index(self, game_index):
        self._validate_index(game_index)
        self.game_index = game_index
        self._switch_active = game_index % 6 >= 3

    def _call_rate(self, quantity, pile_size):
        rate = super()._call_rate(quantity, pile_size)
        if self._switch_active:
            if not getattr(self, "_switch_shadow_plan", False):
                self._switch_call_rate_calls += 1
            raised = max(rate, self.SWITCH_CONFIG["floor"])
            if raised > rate and not getattr(self, "_switch_shadow_plan", False):
                self._switch_floor_raises += 1
            return raised
        return rate

    def _plan(self, hand, state):
        self._switch_call_rate_calls = 0
        self._switch_floor_raises = 0
        self._switch_actual_choice_changes = 0
        shadow = copy(self)
        shadow._switch_active = False
        shadow._switch_shadow_plan = True
        baseline, _ = RoundCenterPolicyBot._plan(shadow, hand, state)
        selected, detail = super()._plan(hand, state)
        changed = selected != baseline
        if changed:
            self._switch_actual_choice_changes += 1
        detail["competitive_switch"] = {
            "active": self._switch_active,
            "normal_choice_changed": changed,
            "floor": self.SWITCH_CONFIG["floor"] if self._switch_active else None,
            "call_rate_calls": self._switch_call_rate_calls,
            "floor_raises": self._switch_floor_raises,
            "actual_choice_changes": int(changed),
        }
        return selected, detail

    def to_dict(self):
        profile = super().to_dict()
        profile["policy"] = self.policy_name
        profile["competitive_switch_config"] = deepcopy(self.SWITCH_CONFIG)
        return profile

    def restore_opponent_profile(self, data):
        if not isinstance(data, dict) or data.get("policy") != self.policy_name:
            raise ValueError("unsupported competitive-switch profile")
        expected = self.to_dict()
        config = data.get("competitive_switch_config")
        if (set(data) != set(expected) or not isinstance(config, dict)
                or set(config) != set(self.SWITCH_CONFIG)
                or type(config.get("policy_id")) is not str
                or config.get("policy_id") != self.policy_name
                or type(config.get("floor")) is not float
                or config.get("floor") != 0.8
                or type(config.get("first_games")) is not list
                or config.get("first_games") != [0, 1, 2]
                or any(type(item) is not int for item in config.get("first_games", []))
                or type(config.get("switched_games")) is not list
                or config.get("switched_games") != [3, 4, 5]
                or any(type(item) is not int for item in config.get("switched_games", []))):
            raise ValueError("competitive-switch profile config mismatch")
        staged_data = deepcopy(data)
        staged_data.pop("competitive_switch_config")
        staged_data["policy"] = RoundCenterPolicyBot.policy_name
        rng = self.rng
        staged = RoundCenterPolicyBot(seed=0)
        staged.restore_opponent_profile(staged_data)
        self.__dict__.update(staged.__dict__)
        self.rng = rng
        self.policy_name = type(self).policy_name
        self.game_index = 0
        self._switch_active = False
        self._switch_call_rate_calls = 0
        self._switch_floor_raises = 0
        self._switch_actual_choice_changes = 0

    def reset(self):
        super().reset()
        self._switch_call_rate_calls = 0
        self._switch_floor_raises = 0
        self._switch_actual_choice_changes = 0


__all__ = ["RoundCompetitiveSwitchActor"]
