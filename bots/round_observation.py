"""The corrected site's player-visible policy input, shared with evaluation."""

from bots.base import public_action_for_viewer, public_actions_for_viewer


def round_player_view(game, viewer: int) -> dict:
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("viewer must be player 0 or 1")
    return {
        "viewer": viewer,
        "player_id": viewer,
        "active_rank": game.active_rank,
        "opponent_hand_size": game.hands[1 - viewer].size(),
        "hand_size": game.hands[viewer].size(),
        "pile_size": len(game.pile),
        "draw_pile_size": len(game.draw_pile),
        "round_number": game.round_number,
        "turn_number": game.turn_count,
        "actions": public_actions_for_viewer(game.actions, viewer),
        "last_action": public_action_for_viewer(game.latest_pending_claim, viewer),
        "own_pile_cards": [
            card for action in game.current_round_actions
            if action.player == viewer for card in action.cards_played
        ],
        "can_play": game.can_play(viewer),
        "can_call_bluff": game.can_call_bluff(viewer),
        "can_pass": game.can_pass(viewer),
    }
