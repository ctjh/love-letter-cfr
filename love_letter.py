"""
Two-player Love Letter as an imperfect-information extensive-form game.

Implements the openCFR `Game` interface. Every predicate also accepts an already
parsed `state=` so solvers can thread state through a traversal instead of
re-parsing the history at every node.

Configuration (read from the environment at import time):
  LL_DECK  full | reduced          which deck to play with (default: full)
  LL_KEY   pr | exact | counts | belief | belief2 | coarse
           the information-set key / abstraction (default: pr)

Actions are integer indices into ACTION_LIST, where each entry is a
(card_played, target, named_card) triple.
"""

import os
from collections import Counter
import numpy as np

from openCFR.games.Game import Game
from openCFR.games.GameNode import GameNode
from openCFR.games.UtilityNode import UtilityNode


# ---------------------------------------------------------------------------
# Deck configuration
# ---------------------------------------------------------------------------
# Standard 2019 deck, two-player rules, without the Chancellor.
FULL_DECK_COUNTS = {
    0: 2,  # Spy
    1: 6,  # Guard
    2: 2,  # Priest
    3: 2,  # Baron
    4: 2,  # Handmaid
    5: 2,  # Prince
    7: 1,  # King
    8: 1,  # Countess
    9: 1,  # Princess
}

# A 6-card deck whose full game tree is small enough for exact CFR and exact
# best-response exploitability. Used to validate the solvers.
REDUCED_DECK_COUNTS = {
    1: 2,  # Guard
    2: 1,  # Priest
    3: 1,  # Baron
    8: 1,  # Countess
    9: 1,  # Princess
}

DECK_COUNTS = (
    REDUCED_DECK_COUNTS
    if os.environ.get("LL_DECK", "full").lower() == "reduced"
    else FULL_DECK_COUNTS
)

FULL_DECK = [c for c, n in DECK_COUNTS.items() for _ in range(n)]
GUARD_TARGETS = [c for c in DECK_COUNTS.keys() if c != 1]  # Guard can't name Guard
CARD_NAMES = {
    0: "Spy", 1: "Guard", 2: "Priest", 3: "Baron", 4: "Handmaid",
    5: "Prince", 7: "King", 8: "Countess", 9: "Princess",
}


# ---------------------------------------------------------------------------
# Utility nodes
# ---------------------------------------------------------------------------
class StaticUtility(UtilityNode):
    def __init__(self, utility):
        self.utility = utility
        super().__init__()

    def get_utility(self):
        return self.utility


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------
# Each action is a (card_played, target, named_card) triple, encoded as a
# single integer index for openCFR's action_map convention.
def _enumerate_actions():
    """All possible action tuples in the game (whether legal or not depends on hand)."""
    actions = []
    seen = set()

    def add(a):
        if a not in seen:
            seen.add(a)
            actions.append(a)

    for card in DECK_COUNTS.keys():
        if card == 1:  # Guard: must name a non-Guard card; target = opponent or None if protected
            for named in GUARD_TARGETS:
                add((1, "opp", named))
            add((1, None, None))  # opponent protected, no effect
        elif card in (2, 3, 7):  # Priest / Baron / King: target opponent or no-op
            add((card, "opp", None))
            add((card, None, None))
        elif card == 5:  # Prince: target self or opponent
            add((5, "self", None))
            add((5, "opp", None))
        else:  # Spy / Handmaid / Countess / Princess: no choice
            add((card, None, None))
    return actions


ACTION_LIST = _enumerate_actions()
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
NUM_ACTIONS = len(ACTION_LIST)


def _action_to_str(a):
    card, target, named = a
    name = CARD_NAMES[card]
    if card == 1 and target == "opp":
        return f"Guard->{CARD_NAMES[named]}"
    if target == "opp":
        return f"{name}->opp"
    if target == "self":
        return f"{name}->self"
    return name


ACTION_MAP = [_action_to_str(a) for a in ACTION_LIST]


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------
def parse_history(history):
    state = {
        "facedown": None,
        "facedown_consumed": False,
        "faceup": None,
        "hands": {0: None, 1: None},
        "drawn": {0: None, 1: None},
        "deck": Counter(FULL_DECK),
        "discard": {0: [], 1: []},
        "eliminated": set(),
        "handmaid_protected": set(),
        "knowledge": {0: None, 1: None},
        "turn": 0,
        "spy_played": {0: False, 1: False},
        # Ordered public action log (Priest's private read is redacted).
        "public_log": [],
        # Per-player ordered private observations (deals, draws, redraws and
        # Priest/King/Baron reveals). Together with public_log this is the
        # player's full action-observation history, i.e. perfect recall.
        "mem": {0: (), 1: ()},
    }

    for event in history:
        if event[0] == "r":
            _apply_chance_event(state, event)
        else:
            _apply_action_event(state, event)
    return state


def _apply_chance_event(state, event):
    _, stage, *args = event

    if stage == "facedown":
        card = args[0]
        state["facedown"] = card
        state["deck"][card] -= 1

    elif stage == "faceup":
        cards = args[0]
        state["faceup"] = cards
        for c in cards:
            state["deck"][c] -= 1

    elif stage == "deal0":
        card = args[0]
        state["hands"][0] = card
        state["deck"][card] -= 1
        state["mem"][0] = state["mem"][0] + (("D", card),)

    elif stage == "deal1":
        card = args[0]
        state["hands"][1] = card
        state["deck"][card] -= 1
        state["mem"][1] = state["mem"][1] + (("D", card),)

    elif stage == "draw":
        player, card = args
        state["drawn"][player] = card
        state["deck"][card] -= 1
        state["mem"][player] = state["mem"][player] + (("d", card),)

    elif stage == "prince_redraw":
        player, card = args
        # The Prince target receives a replacement directly into their hand.
        state["hands"][player] = card
        state["deck"][card] -= 1
        state["mem"][player] = state["mem"][player] + (("r", card),)

    elif stage == "prince_redraw_facedown":
        # Deck was empty; the Prince target draws the face-down set-aside card.
        player, card = args
        state["hands"][player] = card
        state["facedown_consumed"] = True
        state["mem"][player] = state["mem"][player] + (("r", card),)


def _apply_action_event(state, event):
    """An action event is a tuple:
        (player, card_played, target, named_card, effect_result, remaining_card)

    where remaining_card is the card the actor would keep AFTER playing
    card_played (i.e. the non-played card from {held, drawn}). It is included
    so we can correctly resolve Prince-on-self (discard the remaining card).
    """
    player, card_played, target, named_card, effect_result, remaining_card = event
    opponent = 1 - player

    # Public discard pile records this play.
    state["discard"][player].append(card_played)
    state["public_log"].append(
        (player, card_played, target, named_card, effect_result if card_played != 2 else None)
    )

    # Staleness rule: any prior knowledge of a hand that == card_played is now wrong.
    for observer in (0, 1):
        if state["knowledge"][observer] == card_played:
            state["knowledge"][observer] = None

    # By default, the actor keeps remaining_card after playing.
    # Prince-on-self overrides this below.
    state["hands"][player] = remaining_card
    state["drawn"][player] = None

    if card_played == 0:
        state["spy_played"][player] = True

    elif card_played == 1:  # Guard
        if effect_result == "eliminated_opponent":
            state["eliminated"].add(opponent)

    elif card_played == 2:  # Priest
        if target == opponent and effect_result is not None:
            state["knowledge"][player] = effect_result
            # Private observation: the actor permanently saw the opponent's card.
            state["mem"][player] = state["mem"][player] + (("see", effect_result),)

    elif card_played == 3:  # Baron
        if effect_result == "eliminated_opponent":
            state["eliminated"].add(opponent)
        elif effect_result == "eliminated_self":
            state["eliminated"].add(player)
        else:
            # tie: no elimination, but both players saw each other's (equal) card.
            opp_card = state["hands"][opponent]
            state["mem"][player] = state["mem"][player] + (("baronsee", opp_card),)
            state["mem"][opponent] = state["mem"][opponent] + (("baronsee", remaining_card),)

    elif card_played == 4:  # Handmaid
        state["handmaid_protected"].add(player)

    elif card_played == 5:  # Prince
        if target == opponent:
            discarded = state["hands"][opponent]
            if discarded is not None:
                state["discard"][opponent].append(discarded)
            state["hands"][opponent] = None
            state["knowledge"][player] = None
            if discarded == 9:
                state["eliminated"].add(opponent)
        else:
            # Prince on self: the remaining_card we just placed in hand is the
            # one that gets discarded; the actor will redraw via a chance node.
            discarded = remaining_card
            if discarded is not None:
                state["discard"][player].append(discarded)
            state["hands"][player] = None
            if discarded == 9:
                state["eliminated"].add(player)

    elif card_played == 7:  # King: swap hands
        # The actor now holds remaining_card; swap with opponent.
        my_card = state["hands"][player]
        opp_card = state["hands"][opponent]
        state["hands"][player] = opp_card
        state["hands"][opponent] = my_card
        # After the swap, each player knows what they handed the other.
        state["knowledge"][player] = my_card
        state["knowledge"][opponent] = opp_card
        # Private observations: each player permanently saw the other's card.
        state["mem"][player] = state["mem"][player] + (("kingsee", opp_card),)
        state["mem"][opponent] = state["mem"][opponent] + (("kingsee", my_card),)

    elif card_played == 9:  # Princess: self-eliminate
        state["eliminated"].add(player)

    # Advance the turn. Handmaid protection on the *incoming* player expires
    # at the start of their next turn (i.e. now).
    state["handmaid_protected"].discard(opponent)
    state["turn"] = opponent


# ---------------------------------------------------------------------------
# Incremental state advancement
# ---------------------------------------------------------------------------
# parse_history() replays the whole event log, so calling it at every node makes
# a depth-D traversal O(D^2). advance_state() applies a single event to a copy,
# which keeps traversals O(D).
def _copy_state(s):
    return {
        "facedown": s["facedown"],
        "facedown_consumed": s["facedown_consumed"],
        "faceup": s["faceup"],
        "hands": dict(s["hands"]),
        "drawn": dict(s["drawn"]),
        "deck": Counter(s["deck"]),
        "discard": {0: list(s["discard"][0]), 1: list(s["discard"][1])},
        "eliminated": set(s["eliminated"]),
        "handmaid_protected": set(s["handmaid_protected"]),
        "knowledge": dict(s["knowledge"]),
        "turn": s["turn"],
        "spy_played": dict(s["spy_played"]),
        "public_log": list(s["public_log"]),
        "mem": dict(s["mem"]),
    }


def advance_state(state, event):
    """Return a NEW state with `event` applied. Leaves `state` untouched."""
    ns = _copy_state(state)
    if event[0] == "r":
        _apply_chance_event(ns, event)
    else:
        _apply_action_event(ns, event)
    return ns


# ---------------------------------------------------------------------------
# The Game class
# ---------------------------------------------------------------------------
class LoveLetter(Game):

    def __init__(self):
        super().__init__(
            num_actions=NUM_ACTIONS,
            num_players=2,
            action_map=ACTION_MAP,
        )

    # -- node-type predicates ------------------------------------------------

    def is_chance_node(self, history, state=None):
        state = parse_history(history) if state is None else state

        # Setup phase (the empty root has facedown is None, so this also covers it)
        if state["facedown"] is None:
            return True
        if state["faceup"] is None:
            return True

        # A living player has no card and hasn't drawn — either initial deal
        # or a Prince redraw is needed.
        for p in (0, 1):
            if (
                state["hands"][p] is None
                and state["drawn"][p] is None
                and p not in state["eliminated"]
            ):
                return True

        # Normal turn draw: the current player has a hand but no drawn card yet.
        current = state["turn"]
        if (
            current not in state["eliminated"]
            and state["hands"][current] is not None
            and state["drawn"][current] is None
        ):
            return True

        return False

    def is_terminal_node(self, history, state=None):
        state = parse_history(history) if state is None else state

        # The empty root falls through to return False (nobody dealt yet).
        if len(state["eliminated"]) > 0:
            return True

        # Empty deck: round ends when no card can be drawn. We trigger this
        # immediately after a turn has ended (so the current player would now
        # need to draw, but cannot).
        current = state["turn"]
        if (
            current not in state["eliminated"]
            and state["hands"][current] is not None
            and state["drawn"][current] is None
        ):
            remaining = sum(v for v in state["deck"].values() if v > 0)
            if remaining == 0:
                return True

        return False

    # -- chance handling ----------------------------------------------------

    def handle_chance(self, history, state=None, sample=False):
        state = parse_history(history) if state is None else state

        remaining_counts = {c: n for c, n in state["deck"].items() if n > 0}
        total = sum(remaining_counts.values())
        unique = sorted(remaining_counts.keys())

        # Setup: face-down card
        if state["facedown"] is None:
            probs = [remaining_counts[c] / total for c in unique]
            outcomes = [("r", "facedown", c) for c in unique]
            return outcomes, probs

        # Setup: face-up set-aside cards (disabled in this variant).
        if state["faceup"] is None:
            return [("r", "faceup", ())], [1.0]

        # Helper: build a draw distribution over the current deck.
        def draw_dist():
            return unique, [remaining_counts[c] / total for c in unique]

        # If a living player has no card and hasn't drawn, they need a card.
        # Cases: (a) initial deal at game start, (b) Prince forced redraw.
        for p in (0, 1):
            if (
                state["hands"][p] is None
                and state["drawn"][p] is None
                and p not in state["eliminated"]
            ):
                # Initial deal: no actions yet on either side.
                if all(len(state["discard"][q]) == 0 for q in (0, 1)):
                    cards, probs = draw_dist()
                    stage = "deal0" if p == 0 else "deal1"
                    return [("r", stage, c) for c in cards], probs

                # Prince redraw. If the deck is empty, use the face-down card
                # (Love Letter rules: the set-aside card is drawn).
                if total == 0:
                    fd = state["facedown"]
                    assert fd is not None and not state["facedown_consumed"], (
                        "no cards left to draw — game should have ended"
                    )
                    # Deterministic chance outcome.
                    return [("r", "prince_redraw_facedown", p, fd)], [1.0]

                cards, probs = draw_dist()
                return [("r", "prince_redraw", p, c) for c in cards], probs

        # Normal turn draw for the current player.
        current_player = state["turn"]
        cards, probs = draw_dist()
        return [("r", "draw", current_player, c) for c in cards], probs

    # -- terminal utility ---------------------------------------------------

    def get_terminal_utility(self, history, state=None):
        """Return utility from the perspective of the player at the terminal node.

        CFR+ flips signs via utility_multiplier based on next_node.player, so what
        matters is that the value here corresponds to game_node.player (the player
        whose turn it would have been at this state).
        """
        state = parse_history(history) if state is None else state
        # The player at this terminal node = whose turn it would be next.
        viewing_player = state["turn"]

        if len(state["eliminated"]) > 0:
            # Eliminated player loses. If both somehow eliminated (Princess on
            # Prince-self), the actor is the one eliminated.
            eliminated = next(iter(state["eliminated"]))
            winner = 1 - eliminated
            value_for_p0 = 1 if winner == 0 else -1
        else:
            p0 = state["hands"][0]
            p1 = state["hands"][1]
            if p0 is None or p1 is None:
                # Should not happen at a showdown; score it as a draw.
                value_for_p0 = 0
            elif p0 > p1:
                value_for_p0 = 1
            elif p1 > p0:
                value_for_p0 = -1
            else:
                # Tiebreak: higher discard-pile sum wins.
                s0 = sum(state["discard"][0])
                s1 = sum(state["discard"][1])
                if s0 > s1:
                    value_for_p0 = 1
                elif s1 > s0:
                    value_for_p0 = -1
                else:
                    value_for_p0 = 0

        signed = value_for_p0 if viewing_player == 0 else -value_for_p0
        return StaticUtility(signed)

    # -- available actions --------------------------------------------------

    def get_available_actions(self, history, state=None):
        state = parse_history(history) if state is None else state
        current_player = state["turn"]
        opponent = 1 - current_player

        held = state["hands"][current_player]
        drawn = state["drawn"][current_player]
        assert held is not None, "held is None at action time"
        assert drawn is not None, "drawn is None at action time"

        hand = [held, drawn]
        opp_protected = opponent in state["handmaid_protected"]

        # Countess constraint: if you hold Countess with King or Prince, you must play Countess.
        countess_forced = 8 in hand and (5 in hand or 7 in hand)

        actions = []
        for card in set(hand):
            if countess_forced and card != 8:
                continue

            if card == 0:
                actions.append((0, None, None))

            elif card == 1:  # Guard
                if not opp_protected:
                    for named in GUARD_TARGETS:
                        actions.append((1, "opp", named))
                else:
                    actions.append((1, None, None))

            elif card in (2, 3, 7):
                if not opp_protected:
                    actions.append((card, "opp", None))
                else:
                    actions.append((card, None, None))

            elif card == 4:
                actions.append((4, None, None))

            elif card == 5:  # Prince
                actions.append((5, "self", None))
                if not opp_protected:
                    actions.append((5, "opp", None))

            elif card == 8:
                actions.append((8, None, None))

            elif card == 9:
                actions.append((9, None, None))

        # De-duplicate while preserving order.
        seen = set()
        deduped = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                deduped.append(a)

        # Encode as integer indices and return a numpy array (Trainer requires this).
        return np.array([ACTION_TO_IDX[a] for a in deduped])

    # -- player tracking ----------------------------------------------------

    def get_player(self, history, state=None):
        state = parse_history(history) if state is None else state
        return state["turn"]

    # -- infoset key --------------------------------------------------------

    def get_infoset_key(self, history, state=None):
        state = parse_history(history) if state is None else state
        current_player = state["turn"]
        opponent = 1 - current_player

        held = state["hands"][current_player]
        drawn = state["drawn"][current_player]
        # Hand can be (held,) if no drawn card (shouldn't happen at action node).
        if drawn is None:
            hand = (held,) if held is not None else ()
        else:
            hand = tuple(sorted([held, drawn]))

        # Information-set key, selected by LL_KEY:
        #
        #   pr       Perfect recall: hand + public log + full private observation
        #            log. The true game; CFR converges to a Nash equilibrium.
        #   exact    Hand + public log + latest reveal only. Imperfect recall: it
        #            merges histories with different beliefs about the opponent's
        #            card, so exploitability plateaus above zero.
        #   counts   Discard multisets instead of the ordered log.
        #   belief   Hand + exact composition of unseen cards.
        #   belief2  Hand + set of unseen card types + unseen count.
        #   coarse   Hand + game-phase bucket. Smallest; no card counting.
        #
        # All non-pr keys also include Handmaid protection and any known
        # opponent card.
        mode = os.environ.get("LL_KEY", "pr")

        if mode == "pr":
            key = (current_player, hand, state["faceup"],
                   tuple(state["public_log"]), state["mem"][current_player])
            return str(key)

        if mode == "exact":
            key = (current_player, hand, state["faceup"],
                   tuple(state["public_log"]), state["knowledge"][current_player])
            return str(key)

        if mode in ("belief", "belief2"):
            # The opponent's card, the draw pile and the face-down card are all
            # drawn from the multiset of cards this player has not seen, so that
            # multiset summarises their belief (ignoring the order of past plays).
            unseen = Counter(FULL_DECK)
            if held is not None:
                unseen[held] -= 1
            if drawn is not None:
                unseen[drawn] -= 1
            for q in (0, 1):
                for c in state["discard"][q]:
                    unseen[c] -= 1
            if state["faceup"]:
                for c in state["faceup"]:
                    unseen[c] -= 1
            if mode == "belief2":
                unseen_repr = (
                    tuple(sorted(c for c, n in unseen.items() if n > 0)),
                    sum(n for n in unseen.values() if n > 0),
                )
            else:
                unseen_repr = tuple(sorted((c, n) for c, n in unseen.items() if n > 0))
            key = (
                current_player,
                hand,
                unseen_repr,
                opponent in state["handmaid_protected"],
                state["knowledge"][current_player],
            )
            return str(key)

        if mode == "coarse":
            # Phase bucket: exact count for the last 3 draws, then 4-6, then 7+.
            deck_left = sum(n for n in state["deck"].values() if n > 0)
            phase = deck_left if deck_left <= 3 else (4 if deck_left <= 6 else 5)
            key = (
                current_player,
                hand,
                phase,
                opponent in state["handmaid_protected"],
                state["knowledge"][current_player],
            )
            return str(key)

        # mode == "counts"
        my_discards = tuple(sorted(state["discard"][current_player]))
        opp_discards = tuple(sorted(state["discard"][opponent]))
        key = (
            current_player,
            hand,
            state["faceup"],
            my_discards,
            opp_discards,
            opponent in state["handmaid_protected"],
            state["knowledge"][current_player],
        )
        return str(key)

    # -- effect resolution --------------------------------------------------

    def _resolve_effect(self, state, player, card_played, target, named_card, remaining_card):
        """Resolve the result of playing `card_played`. `target` is an integer
        player id (or None), not a string token."""
        opponent = 1 - player

        if card_played == 1:  # Guard
            if target == opponent and state["hands"][opponent] == named_card:
                return "eliminated_opponent"
            return "no_effect"

        if card_played == 2:  # Priest
            if target == opponent:
                return state["hands"][opponent]
            return None

        if card_played == 3:  # Baron
            if target == opponent:
                opp_card = state["hands"][opponent]
                if remaining_card > opp_card:
                    return "eliminated_opponent"
                if opp_card > remaining_card:
                    return "eliminated_self"
                return "tie"
            return None

        if card_played == 5:  # Prince
            if target == opponent:
                if state["hands"][opponent] == 9:
                    return "eliminated_opponent"
                return "opponent_redraws"
            if remaining_card == 9:
                return "eliminated_self"
            return "self_redraws"

        return None

    # -- tree building ------------------------------------------------------

    def build_game_tree(self, history=None):
        if history is None:
            history = []

        if self.is_terminal_node(history):
            return GameNode(
                history,
                self.get_player(history),
                is_terminal_node=True,
                terminal_utility=self.get_terminal_utility(history),
            )

        if self.is_chance_node(history):
            outcomes, probs = self.handle_chance(history)
            children = [self.build_game_tree(history + [o]) for o in outcomes]
            return GameNode(
                history,
                self.get_player(history),
                children,
                is_chance_node=True,
                chance_outcomes=outcomes,
                chance_probs=probs,
            )

        # Action node
        action_indices = self.get_available_actions(history)
        state = parse_history(history)
        current_player = state["turn"]
        held = state["hands"][current_player]
        drawn = state["drawn"][current_player]

        children = []
        for idx in action_indices:
            card_played, target_token, named_card = ACTION_LIST[idx]
            # Decode target token to player id.
            if target_token == "opp":
                target = 1 - current_player
            elif target_token == "self":
                target = current_player
            else:
                target = None

            remaining_card = drawn if card_played == held else held

            effect = self._resolve_effect(
                state, current_player, card_played, target, named_card, remaining_card
            )
            action_event = (current_player, card_played, target, named_card, effect, remaining_card)
            next_history = history + [action_event]
            children.append(self.build_game_tree(next_history))

        return GameNode(history, current_player, children, action_indices)


if __name__ == "__main__":
    print(f"Deck: {DECK_COUNTS}")
    print(f"Actions ({NUM_ACTIONS}): {ACTION_MAP}")
