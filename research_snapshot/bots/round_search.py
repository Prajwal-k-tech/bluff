"""Experimental flat root lookahead; neither a solved policy nor SDMCTS.

Keeps Adaptive's predictor and learning, adds bounded action comparison against
a declared Math continuation model. Shortlist pruning and card-advantage leaves
are hypotheses, not mathematically optimal choices. Not in the web roster.
"""

from collections import Counter
from copy import deepcopy

from bots.round_card_memory import opponent_rank_bounds
from bots.round_policy import RoundAdaptiveBot, RoundMathBot
from bots.round_research import RoundLatentPropensityBot
from bots.round_rollout import compare_root_choices


def action_key(choice):
    return (choice["action"], tuple(choice.get("cards", ())), choice.get("rank"))


class RoundRootRolloutBot(RoundAdaptiveBot):
    policy_name = "round_root_rollout_research"
    samples = 4
    horizon = 8
    stratify_pending = False

    def _shortlist(self, hand, state, baseline):
        """At most11: baseline, best packet per quantity/truth, call and pass.

        Inherits Math's physical-packet representative compression; it is not
        complete for lookahead. Within each quantity/truth bucket the one-step
        heuristic chooses one rank/subset. No opponent identities are read.
        """
        state = dict(state)
        state["_opponent_rank_bounds"] = opponent_rank_bounds(
            state["actions"], self.player_id, state.get("last_action"))
        buckets = {}
        counts = Counter(card.rank for card in hand)
        estimates = {}
        for packet, rank, truthful in self._play_candidates(hand, state):
            quantity = len(packet)
            pile = state["pile_size"] + quantity
            if (rank, quantity) not in estimates:
                estimates[rank, quantity] = max(self._call_rate(quantity, pile),
                    self._certifiable_probability(hand, state, rank, quantity))
            call = 1.0 if quantity == len(hand) else estimates[rank, quantity]
            utility = quantity + call * pile if truthful else quantity - call * pile
            priority = (utility, self._shape_tiebreak(packet, counts))
            key = (quantity, truthful)
            if key not in buckets or priority > buckets[key][0]:
                buckets[key] = (priority, self._play(list(packet), rank))
        choices = [baseline] + [value[1] for value in buckets.values()]
        if state.get("can_call_bluff"):
            choices.append(self._simple("call_bluff"))
        if state.get("can_pass"):
            choices.append(self._simple("pass"))
        unique = {}
        for choice in choices:
            unique.setdefault(action_key(choice), choice)
        return list(unique.values())

    def _focal_continuation_profile(self):
        return self.to_dict()

    def _opponent_continuation(self):
        """Default model unchanged; research ablations may override explicitly."""
        return RoundMathBot(0)

    def choose_action(self, hand, state):
        baseline = super().choose_action(hand, state)
        if self.last_decision_telemetry.get("reason") == "truthful_terminal_finish":
            self.last_decision_telemetry["search_skipped"] = "guaranteed_win"
            return baseline
        choices = self._shortlist(hand, state, baseline)
        # Plain Adaptive continuations avoid recursive search and inherit only
        # the focal player's legal evidence, not the opponent's private profile.
        continuation = RoundAdaptiveBot()
        continuation.restore_opponent_profile({**self._focal_continuation_profile(),
                                                "policy": continuation.policy_name})
        continuation.rng.setstate(self.rng.getstate())
        continuation._resolved_action_objects = deepcopy(self._resolved_action_objects)
        options = {"stratify_pending": True} if self.stratify_pending else {}
        opponent_continuation = self._opponent_continuation()
        comparison = compare_root_choices(continuation, opponent_continuation, hand, state,
            choices, samples=self.samples, horizon=self.horizon,
            seed=self.rng.randrange(2 ** 63), **options)
        selected = max(range(len(choices)),
            key=lambda i: comparison["scores"][i]["mean_heuristic_value"])
        self.last_decision_telemetry = {
            "policy": self.policy_name, "action": choices[selected]["action"],
            "baseline_action": baseline["action"], "search": "flat_root_monte_carlo",
            "opponent_continuation": opponent_continuation.policy_name, "shortlist_size": len(choices),
            "selected_index": selected, **comparison,
        }
        return choices[selected]


class RoundStratifiedRolloutBot(RoundRootRolloutBot):
    """Same research search with weighted latest-truth strata, not a new prior."""

    policy_name = "round_stratified_rollout_research"
    stratify_pending = True


class RoundLatentCallSearchBot(RoundLatentPropensityBot):
    """Compare the latent baseline only with calling the pending claim.

    This research diagnostic targets strategic challenges the baseline misses;
    rollout leaves remain the existing terminal/card-advantage heuristic.
    """

    policy_name = "round_latent_call_search_research"
    samples = 4
    horizon = 64

    def choose_action(self, hand, state):
        baseline = super().choose_action(hand, state)
        baseline_telemetry = deepcopy(self.last_decision_telemetry)
        reason = baseline_telemetry.get("reason")
        if reason == "truthful_terminal_finish":
            baseline_telemetry["search_skipped"] = "guaranteed_win"
            self.last_decision_telemetry = baseline_telemetry
            return baseline
        if not state.get("can_call_bluff"):
            baseline_telemetry["search_skipped"] = "cannot_call"
            self.last_decision_telemetry = baseline_telemetry
            return baseline
        if baseline.get("action") == "call_bluff":
            baseline_telemetry["search_skipped"] = "baseline_already_calls"
            self.last_decision_telemetry = baseline_telemetry
            return baseline

        choices = [baseline, self._simple("call_bluff")]
        continuation = RoundLatentPropensityBot()
        continuation.restore_opponent_profile({
            **self.to_dict(), "policy": RoundLatentPropensityBot.policy_name,
        })
        continuation.player_id = self.player_id
        continuation.rng.setstate(self.rng.getstate())
        continuation._resolved_action_objects = deepcopy(self._resolved_action_objects)
        continuation._latent_forecasts = deepcopy(self._latent_forecasts)
        comparison = compare_root_choices(
            continuation, RoundMathBot(0), hand, state, choices,
            samples=self.samples, horizon=self.horizon,
            seed=self.rng.randrange(2 ** 63), stratify_pending=True,
        )
        scores = comparison["scores"]
        selected = (1 if scores[1]["mean_heuristic_value"]
                    > scores[0]["mean_heuristic_value"] else 0)
        self.last_decision_telemetry = {
            **baseline_telemetry,
            "policy": self.policy_name,
            "action": choices[selected]["action"],
            "baseline_action": baseline["action"],
            "search": "latent_call_root_comparison",
            "opponent_continuation": "round_math_approximation",
            "selected_index": selected,
            "choices": tuple(choice["action"] for choice in choices),
            "samples": self.samples,
            "horizon": self.horizon,
            "opponent_model": "round_math_approximation",
            "comparison": comparison,
        }
        return choices[selected]

    def restore_opponent_profile(self, data):
        if not isinstance(data, dict) or data.get("policy") != self.policy_name:
            raise ValueError("unsupported latent call-search profile")
        candidate = RoundLatentPropensityBot()
        candidate.restore_opponent_profile({
            **data, "policy": RoundLatentPropensityBot.policy_name,
        })
        self._call_evidence = candidate._call_evidence
        self._bluff_evidence = candidate._bluff_evidence
        self._latent_weights = candidate._latent_weights
        self._latent_updates = candidate._latent_updates
        self._latent_forecasts.clear()
