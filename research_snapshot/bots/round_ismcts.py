"""Research-only, bounded single-observer information-set tree search.

This is an approximate focal-player IS-MCTS prototype, not SDMCTS, an
optimal policy, or a calibrated belief model. It uses constrained root
determinizations, approximate continuation policies, representative action
pruning, and a bounded card-advantage horizon value.

The optional ``rank_groups`` leaf proxy values fewer distinct held ranks. It is
an optimistic truthful-shedding-action proxy, not expected turns or win
probability: it assumes rank groups can be shed in packets of up to four and
omits pass/draw, challenge, initiative, and bluff costs.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import math
import random
from numbers import Real

from bots.base import public_action_for_viewer
from bots.round_determinization import sample_root_world
from bots.round_observation import round_player_view
from bots.round_rollout import (apply_choice, materialize_root,
                                prepare_injected_forecast)
from bots.round_policy import RoundAdaptiveBot
from bots.round_search import RoundLatentCallSearchBot, RoundRootRolloutBot
from bots.round_opening_likelihood import STYLES
from bots.round_generative_style import RoundGenerativeStyleBot
from bots.round_joint_policy_root import (
    HYPOTHESES as JOINT_HYPOTHESES,
    actor_for_joint_root,
    sample_joint_policy_root,
)
from bots.round_policy_mixture_actor import RoundPolicyMixtureActor
from cards import Card


def information_state_key(hand, view):
    """Canonical focal observation; never includes sampled hidden identities."""
    viewer = view.get("player_id", view.get("viewer"))
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("invalid focal observation")

    def card_key(card):
        if not isinstance(card, Card):
            raise ValueError("information-state cards must be visible Card values")
        return int(card.rank), int(card.suit)

    def action_key(action):
        player = action.player
        own = player == viewer
        return (
            player, action.action_index, action.round_number,
            (tuple(card_key(card) for card in action.cards_played)
             if own or action.bluff_called
             or getattr(action, "cards_revealed", False) is True else
             tuple(None for _ in action.cards_played)),
            int(action.claimed_rank), action.bluff_called,
            action.caller_was_right if action.bluff_called else None,
            action.response_was_call, action.response_was_forced,
            action.response_player, action.response_kind,
            action.pile_size_before,
            getattr(action, "cards_revealed", False) is True,
        )

    last = view.get("last_action")
    last_key = None if last is None else action_key(last)
    return (
        viewer,
        tuple(sorted(card_key(card) for card in hand)),
        None if view.get("active_rank") is None else int(view["active_rank"]),
        view.get("round_number"), view.get("turn_number"),
        view.get("hand_size"), view.get("opponent_hand_size"),
        view.get("pile_size"), view.get("draw_pile_size"),
        bool(view.get("can_play")), bool(view.get("can_pass")),
        bool(view.get("can_call_bluff")),
        tuple(action_key(action) for action in view.get("actions", ())),
        last_key,
        tuple(sorted(card_key(card) for card in view.get("own_pile_cards", ()))),
        tuple(tuple((packet["action_index"], int(packet["claimed_rank"]),
                     tuple(card_key(card) for card in packet["cards"]))
                    for packet in pickup)
              for pickup in view.get("private_pickups", ())),
    )


def _action_key(choice):
    cards = tuple(sorted((int(card.rank), int(card.suit))
                         for card in choice.get("cards", ())))
    rank = choice.get("rank")
    return choice.get("action", choice.get("kind")), cards, None if rank is None else int(rank)


def _validated_style_weights(weights):
    if weights is None:
        return None
    if not isinstance(weights, dict) or set(weights) != set(STYLES):
        raise ValueError("opponent_style_weights must map exactly the canonical styles")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("opponent style weights must be finite nonnegative numbers")
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("opponent style weights must have positive finite sum")
    return {style: weights[style] / total for style in STYLES}


def _validated_joint_weights(weights):
    if not isinstance(weights, dict):
        raise ValueError("joint_policy_weights must be normalized over four hypotheses")
    try:
        # Reuse the frozen actor's strict key/type/normalization contract.
        return dict(RoundPolicyMixtureActor(weights, seed=0).weights)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "joint_policy_weights must be normalized over four hypotheses") from error


def _validated_joint_truth(probabilities, *, pending):
    if pending is None:
        if probabilities is not None:
            raise ValueError("joint_truth_probabilities require a pending claim")
        return None
    if (not isinstance(probabilities, dict)
            or set(probabilities) != set(JOINT_HYPOTHESES)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in probabilities.values())
            or probabilities["Honest"] != 1.):
        raise ValueError("pending joint root requires valid per-hypothesis truth laws with Honest fixed true")
    return dict(probabilities)


def _preflight_joint_support(hand, view, weights, truth_probabilities):
    """Reject every positive-mass policy/truth stratum lacking world support."""
    pending = view.get("last_action")
    for hypothesis, weight in weights.items():
        if weight <= 0:
            continue
        one_hot = {name: float(name == hypothesis) for name in JOINT_HYPOTHESES}
        if pending is None:
            sample_joint_policy_root(
                hand, view, random.Random(0), weights=one_hot,
                truth_probabilities=None)
            continue
        probability = truth_probabilities[hypothesis]
        strata = ((True, probability), (False, 1.0 - probability))
        for truth, mass in strata:
            if mass <= 0:
                continue
            conditional = dict(truth_probabilities)
            conditional[hypothesis] = float(truth)
            sample_joint_policy_root(
                hand, view, random.Random(0), weights=one_hot,
                truth_probabilities=conditional)


@dataclass
class _Edge:
    visits: int = 0
    value_sum: float = 0.0
    available: int = 0


@dataclass
class _Node:
    visits: int = 0
    edges: dict = field(default_factory=dict)


def _heuristic_value(game, viewer):
    return (game.hands[1 - viewer].size() - game.hands[viewer].size()) / 52.0


def _leaf_value(game, viewer, leaf_value):
    if leaf_value == "terminal_neutral":
        # Traversal surrogate only: unresolved games remain unknown in telemetry.
        return 0.0
    if leaf_value == "cards":
        return _heuristic_value(game, viewer)
    if leaf_value == "rank_groups":
        own_groups = len({card.rank for card in game.hands[viewer].cards})
        opponent_groups = len({card.rank for card in game.hands[1 - viewer].cards})
        return (opponent_groups - own_groups) / 13.0
    raise ValueError("unsupported leaf value")


def _advance_search_branch(game, policies, choice, *, tree_selected=False):
    """Match arena observations; replace hypothetical baseline forecast only.

    Tree choices bypass choose_action or replace its baseline choice. Ordinary
    rollout choices already register their own forecasts. Frozen/count-only
    continuations explicitly opt out of response-expert registration.
    """
    actor = game.current_player
    policy = policies[actor]
    if tree_selected:
        view = round_player_view(game, actor)
        prepare_injected_forecast(policy, list(game.hands[actor].cards), view, choice)
    events = apply_choice(game, choice)
    for action in events:
        for seat, observer in enumerate(policies):
            observer.observe_action(public_action_for_viewer(action, seat),
                                    game.hands[1-seat].size())
    for seat, observer in enumerate(policies):
        observe_view = getattr(observer, "observe_player_view", None)
        if callable(observe_view):
            observe_view(round_player_view(game, seat))


def _recommend_root_action(root_visits, totals, root_recommendation="mean"):
    """Select a root action, preserving insertion order for exact ties."""
    if root_recommendation == "mean":
        # Keep the original key and tuple ordering for default parity.
        return max(root_visits, key=lambda key: (
            totals[key][1] / totals[key][0] if totals[key][0] else -math.inf,
            totals[key][0]))
    if root_recommendation == "visits":
        return max(root_visits, key=lambda key: (
            root_visits[key],
            totals[key][1] / totals[key][0] if totals[key][0] else -math.inf))
    raise ValueError("root_recommendation must be 'mean' or 'visits'")


def _has_truthful_finish_opportunity(hand, view):
    """Whether the actor's legal observation admits a truthful emptying play."""
    if not view.get("can_play") or not 1 <= len(hand) <= 4:
        return False
    active_rank = view.get("active_rank")
    counts = {}
    for card in hand:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return any(count == len(hand) and
               (active_rank is None or rank == active_rank)
               for rank, count in counts.items())


def _focal_candidates(policy, hand, view):
    # Existing math/adaptive shortlist intentionally compresses physical
    # packets by quantity/truth family; it is not a complete action space.
    baseline = policy.choose_action(list(hand), view)
    return RoundRootRolloutBot._shortlist(policy, list(hand), view, baseline)


def _widened_descendant_actions(legal_keys, node):
    """Keep visited legal edges and progressively admit baseline-first options."""
    cap = max(1, math.isqrt(node.visits + 1))
    active = [key for key in legal_keys if node.edges[key].visits > 0]
    active_set = set(active)
    for key in legal_keys:
        if len(active) >= cap:
            break
        if key not in active_set:
            active.append(key)
            active_set.add(key)
    return active, cap


def search_information_set(policy, opponent_model, hand, state, choices,
                           iterations=16, horizon=12, seed=0, max_nodes=128,
                           leaf_value="cards", opponent_style_weights=None,
                           pending_bluff_probability=None,
                           mixed_respects_certificates=False,
                           mixed_takes_truthful_finish=False,
                           finish_extension=0,
                           round_boundary_extension=0,
                           round_boundary_extension_scope=0,
                           opponent_call_probability=0.35,
                           root_recommendation="mean",
                           joint_policy_weights=None,
                           joint_truth_probabilities=None,
                           tree_policy="information_tree",
                           descendant_widening=False):
    """Search choices at a focal observation via bounded IS-MCTS traversal.

    `choices` is the caller's representative root shortlist. Each iteration
    samples a fresh constrained world; pending truth is sampled from one frozen
    approximate pre-search posterior unless the opt-in coherent joint-root
    proposal is supplied. Only focal-seat decision nodes are
    stored. Opponent actions come from their observation-limited continuation
    model, never UCT over the sampled hidden state. Optional style weights
    sample one independent future-continuation actor per iteration; this is an
    approximate policy mixture, not a joint history posterior or SDMCTS.
    ``finish_extension`` optionally extends a bounded leaf by at most two
    policy-selected transitions when the simulated actor has a legal truthful
    emptying packet; zero preserves the ordinary horizon.
    ``round_boundary_extension`` continues an open round through sampled
    policy decisions. Scope 0 preserves the original root-round behavior;
    scope 1 follows the round open at the bounded leaf. The two extension
    modes are exclusive.
    ``root_bandit`` keeps UCT at the root but uses the same continuation policy
    below it, isolating deeper tree search under the same model/budget. The
    terminal_neutral leaf surrogate is zero for traversal, NOT an eventual-win
    estimate or a relabeling of cutoffs as draws.
    """
    viewer = state.get("player_id", state.get("viewer"))
    if (type(viewer) is not int or viewer not in (0, 1)
            or type(iterations) is not int or iterations < 1
            or type(horizon) is not int or horizon < 1
            or type(max_nodes) is not int or max_nodes < 1
            or type(finish_extension) is not int or not 0 <= finish_extension <= 2
            or type(round_boundary_extension) is not int
            or not 0 <= round_boundary_extension <= 8
            or type(round_boundary_extension_scope) is not int
            or round_boundary_extension_scope not in (0, 1)
            or (round_boundary_extension_scope and not round_boundary_extension)
            or (finish_extension and round_boundary_extension)
            or iterations * (horizon + finish_extension + round_boundary_extension) > 10000
            or type(seed) is not int
            or type(leaf_value) is not str
            or leaf_value not in ("cards", "rank_groups", "terminal_neutral")
            or tree_policy not in ("information_tree", "root_bandit")
            or not isinstance(choices, (list, tuple)) or not choices
            or len(choices) > 64
            or type(root_recommendation) is not str
            or root_recommendation not in ("mean", "visits")):
        raise ValueError("invalid information-set search configuration or budget")
    if type(mixed_respects_certificates) is not bool:
        raise TypeError("mixed_respects_certificates must be a bool")
    if type(mixed_takes_truthful_finish) is not bool:
        raise TypeError("mixed_takes_truthful_finish must be a bool")
    if type(descendant_widening) is not bool:
        raise TypeError("descendant_widening must be a bool")
    if descendant_widening and tree_policy != "information_tree":
        raise ValueError("descendant_widening requires information_tree")
    if (isinstance(opponent_call_probability, bool)
            or not isinstance(opponent_call_probability, Real)
            or not math.isfinite(opponent_call_probability)
            or not 0 <= opponent_call_probability <= 1):
        raise ValueError("opponent_call_probability must be finite and in [0, 1]")
    opponent_call_probability = float(opponent_call_probability)
    joint_mode = (joint_policy_weights is not None
                  or joint_truth_probabilities is not None)
    if joint_mode:
        if joint_policy_weights is None:
            raise ValueError("joint_truth_probabilities require joint_policy_weights")
        joint_policy_weights = _validated_joint_weights(joint_policy_weights)
        joint_truth_probabilities = _validated_joint_truth(
            joint_truth_probabilities, pending=state.get("last_action"))
        if (opponent_style_weights is not None
                or pending_bluff_probability is not None
                or mixed_respects_certificates
                or mixed_takes_truthful_finish
                or opponent_call_probability != 0.35):
            raise ValueError("joint policy roots are exclusive with legacy style/truth/mixed controls")
    elif joint_truth_probabilities is not None:
        # Kept explicit even though joint_mode catches this case.
        raise ValueError("joint_truth_probabilities require joint_policy_weights")
    if mixed_respects_certificates and opponent_style_weights is None:
        raise ValueError("mixed_respects_certificates requires opponent_style_weights")
    if mixed_takes_truthful_finish and opponent_style_weights is None:
        raise ValueError("mixed_takes_truthful_finish requires opponent_style_weights")
    if opponent_call_probability != 0.35 and opponent_style_weights is None:
        raise ValueError("nondefault opponent_call_probability requires opponent_style_weights")
    if (isinstance(policy, (RoundRootRolloutBot, RoundLatentCallSearchBot))
            or isinstance(opponent_model, (RoundRootRolloutBot, RoundLatentCallSearchBot))):
        raise ValueError("search requires non-search continuation policies")
    hand = tuple(hand)
    root_view = deepcopy(state)
    root_policy = deepcopy(policy)
    root_policy.player_id = viewer
    pending = root_view.get("last_action")
    if joint_mode:
        _preflight_joint_support(
            hand, root_view, joint_policy_weights, joint_truth_probabilities)
    bluff_probability = None
    if not joint_mode and pending_bluff_probability is not None:
        if (isinstance(pending_bluff_probability, bool)
                or not isinstance(pending_bluff_probability, (int, float))
                or not math.isfinite(pending_bluff_probability)
                or not 0 <= pending_bluff_probability <= 1 or pending is None):
            raise ValueError("explicit pending bluff probability requires a pending claim and a finite value in [0, 1]")
        bluff_probability = float(pending_bluff_probability)
    elif not joint_mode and pending is not None:
        bluff_probability, _ = root_policy._posterior_bluff(hand, root_view, pending)
        if not math.isfinite(bluff_probability) or not 0 <= bluff_probability <= 1:
            raise ValueError("invalid pending-claim forecast")

    rng = random.Random(seed)
    normalized_style_weights = _validated_style_weights(opponent_style_weights)
    sampled_style_counts = {style: 0 for style in STYLES} if normalized_style_weights is not None else None
    sampled_joint_policy_counts = ({hypothesis: 0 for hypothesis in JOINT_HYPOTHESES}
                                   if joint_mode else None)
    sampled_joint_truth_counts = ({hypothesis: {"truthful": 0, "false": 0,
                                                "not_applicable": 0}
                                  for hypothesis in JOINT_HYPOTHESES}
                                 if joint_mode else None)
    sampled_joint_root_plan_counts = {} if joint_mode else None
    nodes = {}
    root_key = information_state_key(hand, root_view)
    root_visits = {key: 0 for key in map(_action_key, choices)}
    totals = {key: [0, 0.0] for key in root_visits}
    outcome_components = {key: {
        "completed_wins": 0, "completed_losses": 0,
        "horizon_leaves": 0, "leaf_value_sum": 0.0,
        "terminal_value_sum": 0.0, "sum_squared_value": 0.0,
    } for key in root_visits}
    root_joint_plan_counts = {key:{} for key in root_visits} if joint_mode else None
    transitions = leaves = actual_extension_transitions = 0
    descendant_uct_selections = descendant_unvisited_selections = 0
    descendant_widening_cap_blocked_actions = 0
    boundary_resolved = boundary_capped = boundary_actual_transitions = 0
    caller_random_state = random.getstate()
    try:
        for iteration in range(iterations):
            if joint_mode:
                joint_root = sample_joint_policy_root(
                    hand, root_view, rng, weights=joint_policy_weights,
                    truth_probabilities=joint_truth_probabilities)
                hypothesis = joint_root.hypothesis
                truth = joint_root.pending_truth
                world = joint_root.world
                sampled_joint_policy_counts[hypothesis] += 1
                truth_label = ("not_applicable" if truth is None else
                               "truthful" if truth else "false")
                sampled_joint_truth_counts[hypothesis][truth_label] += 1
                plan_key = f"{hypothesis}:{truth_label}"
                sampled_joint_root_plan_counts[plan_key] = (
                    sampled_joint_root_plan_counts.get(plan_key, 0) + 1)
            else:
                # Preserve the legacy root proposal and its RNG draw sequence.
                # sample_root_world uses True to mean truthful, not bluffing.
                truth = None if pending is None else rng.random() >= bluff_probability
                world = sample_root_world(hand, root_view, rng, pending_truth=truth)
            game = materialize_root(hand, root_view, world)
            # Carry the frozen pre-reveal forecast into each isolated branch
            # so a simulated challenge can update the continuation learner.
            focal = deepcopy(root_policy)
            if joint_mode:
                opponent = actor_for_joint_root(
                    joint_root, seed=rng.randrange(2 ** 63))
            elif normalized_style_weights is None:
                opponent = deepcopy(opponent_model)
            else:
                style_pick = rng.random()
                cumulative = 0.0
                sampled_style = STYLES[-1]
                for style in STYLES:
                    cumulative += normalized_style_weights[style]
                    if style_pick < cumulative:
                        sampled_style = style
                        break
                sampled_style_counts[sampled_style] += 1
                opponent = RoundGenerativeStyleBot(
                    sampled_style, seed=rng.randrange(2 ** 63),
                    call_probability=opponent_call_probability,
                    mixed_respects_certificates=mixed_respects_certificates,
                    mixed_takes_truthful_finish=mixed_takes_truthful_finish)
            policies = [None, None]
            policies[viewer], policies[1 - viewer] = focal, opponent
            for seat, actor_policy in enumerate(policies):
                actor_policy.player_id = seat
            # Engine pass/recycle shuffling uses process-global RNG. Seed each
            # isolated branch and restore the caller's state around the search.
            random.seed(rng.randrange(2 ** 63))
            path = []
            expanded = False
            for decision in range(horizon):
                if game.game_over:
                    break
                actor = game.current_player
                actor_hand = list(game.hands[actor].cards)
                actor_view = round_player_view(game, actor)
                tree_selected = actor == viewer and (decision == 0 or tree_policy == "information_tree")
                if tree_selected:
                    candidates = (list(choices) if decision == 0 else
                                  _focal_candidates(focal, actor_hand, actor_view))
                    key = information_state_key(actor_hand, actor_view)
                    if decision == 0 and key != root_key:
                        raise ValueError("sampled root changed the focal observation")
                    node = nodes.get(key)
                    if node is None and len(nodes) < max_nodes:
                        node = nodes.setdefault(key, _Node())
                    if node is not None:
                        legal = {_action_key(choice): choice for choice in candidates}
                        for action_key in legal:
                            node.edges.setdefault(action_key, _Edge())
                            node.edges[action_key].available += 1
                        available = list(legal)
                        if descendant_widening and decision > 0:
                            active, _ = _widened_descendant_actions(available, node)
                            descendant_widening_cap_blocked_actions += (
                                len(available) - len(active))
                            available = active
                        unvisited = [ak for ak in available if node.edges[ak].visits == 0]
                        if unvisited:
                            selected = unvisited[0]
                            descendant_unvisited_selections += decision > 0
                        else:
                            descendant_uct_selections += decision > 0
                            selected = max(available, key=lambda ak:
                                node.edges[ak].value_sum / node.edges[ak].visits
                                + math.sqrt(2 * math.log(max(1, node.visits))
                                             / node.edges[ak].visits))
                        edge = node.edges[selected]
                        choice = legal[selected]
                        path.append((key, selected))
                        if edge.visits == 0 and not expanded:
                            expanded = True
                        if decision == 0:
                            root_visits[selected] += 1
                    else:
                        choice = candidates[0]
                else:
                    choice = policies[actor].choose_action(actor_hand, actor_view)
                _advance_search_branch(game, policies, choice, tree_selected=tree_selected)
                transitions += 1
                # Once one edge is expanded, continuation remains policy based.
                if expanded:
                    for _ in range(decision + 1, horizon):
                        if game.game_over:
                            break
                        actor = game.current_player
                        view = round_player_view(game, actor)
                        _advance_search_branch(game, policies, policies[actor].choose_action(
                            list(game.hands[actor].cards), view))
                        transitions += 1
                    break
            # At a bounded leaf, spend only the requested tactical extension
            # when the sampled actor's own legal view admits a truthful finish.
            # The actor still chooses through its ordinary continuation model.
            extension_transitions = 0
            while (not game.game_over and
                   extension_transitions < finish_extension):
                actor = game.current_player
                actor_hand = list(game.hands[actor].cards)
                actor_view = round_player_view(game, actor)
                if not _has_truthful_finish_opportunity(actor_hand, actor_view):
                    break
                _advance_search_branch(
                    game, policies, policies[actor].choose_action(actor_hand, actor_view))
                transitions += 1
                extension_transitions += 1
            actual_extension_transitions += extension_transitions
            boundary_extension_transitions = 0
            if (round_boundary_extension and not game.game_over
                    and game.active_rank is not None):
                boundary_round = (root_view.get("round_number")
                                  if round_boundary_extension_scope == 0
                                  else game.round_number)
                if game.round_number != boundary_round:
                    boundary_round = None
            else:
                boundary_round = None
            if boundary_round is not None:
                while (not game.game_over
                       and game.round_number == boundary_round
                       and boundary_extension_transitions < round_boundary_extension):
                    actor = game.current_player
                    actor_hand = list(game.hands[actor].cards)
                    actor_view = round_player_view(game, actor)
                    _advance_search_branch(
                        game, policies, policies[actor].choose_action(actor_hand, actor_view))
                    transitions += 1
                    boundary_extension_transitions += 1
                boundary_actual_transitions += boundary_extension_transitions
                if game.game_over or game.round_number != boundary_round:
                    boundary_resolved += 1
                elif boundary_extension_transitions == round_boundary_extension:
                    boundary_capped += 1
            terminal = game.game_over
            if terminal:
                value = 1.0 if game.winner == viewer else -1.0
            else:
                value = _leaf_value(game, viewer, leaf_value)
                leaves += 1
            for key, action_key in path:
                node = nodes[key]
                edge = node.edges[action_key]
                node.visits += 1
                edge.visits += 1
                edge.value_sum += value
            if not path:
                raise RuntimeError("root action was not represented in the tree")
            root_action = path[0][1]
            if joint_mode:
                counts = root_joint_plan_counts[root_action]
                counts[plan_key] = counts.get(plan_key,0)+1
            totals[root_action][0] += 1
            totals[root_action][1] += value
            component = outcome_components[root_action]
            if terminal:
                if game.winner == viewer:
                    component["completed_wins"] += 1
                else:
                    component["completed_losses"] += 1
                component["terminal_value_sum"] += value
            else:
                component["horizon_leaves"] += 1
                component["leaf_value_sum"] += value
            component["sum_squared_value"] += value * value
    finally:
        random.setstate(caller_random_state)

    from bots.round_terminal_bounds import terminal_completion_bounds
    selected = _recommend_root_action(root_visits, totals, root_recommendation)
    result = {
        "choice": next(choice for choice in choices if _action_key(choice) == selected),
        "telemetry": {
            "algorithm": ("approximate_focal_information_set_mcts" if tree_policy == "information_tree"
                          else "root_uct_bandit_policy_continuation"),
            "tree_policy": tree_policy,
            "descendant_widening": descendant_widening,
            "descendant_widening_cap_blocked_actions": descendant_widening_cap_blocked_actions,
            "descendant_uct_selections": descendant_uct_selections,
            "descendant_unvisited_selections": descendant_unvisited_selections,
            "descendant_revisited_nodes": sum(n.visits > 1 for k, n in nodes.items() if k != root_key),
            "leaf_value": leaf_value,
            "iterations": iterations, "nodes": len(nodes),
            "root_recommendation": root_recommendation,
            "transitions": transitions, "leaf_heuristic_count": leaves,
            "finish_extension_requested": finish_extension,
            "finish_extension_transitions": actual_extension_transitions,
            "round_boundary_extension_requested": round_boundary_extension,
            "round_boundary_extension_scope": round_boundary_extension_scope,
            "round_boundary_extension_transitions": boundary_actual_transitions,
            "round_boundary_extension_resolved": boundary_resolved,
            "round_boundary_extension_capped_open": boundary_capped,
            "root_visits": {_action_key(choice): root_visits[_action_key(choice)]
                            for choice in choices},
            "root_mean_heuristic_value": {
                key: (totals[key][1] / totals[key][0] if totals[key][0] else None)
                for key in totals},
            "root_outcome_components": outcome_components,
            "root_empirical_completion_bounds": {
                key: terminal_completion_bounds(c["completed_wins"],
                    c["completed_losses"], c["horizon_leaves"])
                for key, c in outcome_components.items()},
            "pending_bluff_forecast": bluff_probability,
            "pending_forecast_source": (
                "joint_conditional_policy_law" if joint_mode else
                "explicit" if pending_bluff_probability is not None else
                "root_policy_posterior" if pending is not None else "none"),
            "opponent_style_weights": normalized_style_weights,
            "opponent_call_probability": opponent_call_probability,
            "sampled_opponent_style_counts": sampled_style_counts,
            "mixed_respects_certificates": mixed_respects_certificates,
            "representative_action_pruning": True,
            "belief_model": (
                "caller_supplied_conditional_policy_and_truth_laws_plus_constrained_root_support_not_full_history_posterior"
                if joint_mode else "approximate_root_constraints_and_frozen_forecast"),
        },
    }
    if joint_mode:
        result["telemetry"].update({
            "sampled_joint_policy_counts": sampled_joint_policy_counts,
            "sampled_joint_truth_counts": sampled_joint_truth_counts,
            "sampled_joint_root_plan_counts": sampled_joint_root_plan_counts,
            "root_action_joint_plan_counts": root_joint_plan_counts,
            "joint_root_provenance": {
                "source": "caller_supplied_conditional_policy_and_truth_laws",
                "full_behavioral_history_posterior": False,
                "policy_weights": dict(joint_policy_weights),
                "conditional_truth_probabilities": (
                    None if joint_truth_probabilities is None else
                    dict(joint_truth_probabilities)),
                "unknown_actor_model": "Random fallback",
            },
        })
    if mixed_takes_truthful_finish:
        result["telemetry"]["mixed_takes_truthful_finish"] = True
    return result
