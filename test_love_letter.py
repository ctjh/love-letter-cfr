"""Rule tests for the Love Letter game logic. Run with `pytest` or directly."""
import love_letter as LL
from love_letter import parse_history, LoveLetter


def _set_deck(counts):
    """Rebuild the module-level deck and action tables for `counts`."""
    LL.DECK_COUNTS = counts
    LL.FULL_DECK = [c for c, n in counts.items() for _ in range(n)]
    LL.GUARD_TARGETS = [c for c in counts if c != 1]
    LL.ACTION_LIST = LL._enumerate_actions()
    LL.ACTION_TO_IDX = {a: i for i, a in enumerate(LL.ACTION_LIST)}
    LL.NUM_ACTIONS = len(LL.ACTION_LIST)
    LL.ACTION_MAP = [LL._action_to_str(a) for a in LL.ACTION_LIST]


_ORIGINAL_DECK = LL.DECK_COUNTS


def test_guard_correct_guess_eliminates():
    """Player 0 plays Guard naming Priest; player 1 holds Priest is eliminated."""
    history = [
        ("r", "facedown", 1),
        ("r", "faceup", ()),
        ("r", "deal0", 1),   # P0 holds Guard
        ("r", "deal1", 2),   # P1 holds Priest
        ("r", "draw", 0, 9), # P0 draws Princess
        # P0 plays Guard naming Priest, keeps Princess.
        (0, 1, 1, 2, "eliminated_opponent", 9),
    ]
    state = parse_history(history)
    assert 1 in state["eliminated"], f"P1 should be eliminated, got {state}"


def test_guard_wrong_guess_no_effect():
    history = [
        ("r", "facedown", 1),
        ("r", "faceup", ()),
        ("r", "deal0", 1),
        ("r", "deal1", 8),   # P1 holds Countess, not Priest
        ("r", "draw", 0, 9),
        (0, 1, 1, 2, "no_effect", 9),
    ]
    state = parse_history(history)
    assert 1 not in state["eliminated"]
    assert state["turn"] == 1
    assert state["hands"][0] == 9  # P0 kept Princess


def test_baron_higher_wins():
    """P0 plays Baron with Princess remaining vs P1's Countess: P1 is eliminated."""
    history = [
        ("r", "facedown", 1),
        ("r", "faceup", ()),
        ("r", "deal0", 3),   # Baron
        ("r", "deal1", 8),   # Countess
        ("r", "draw", 0, 9), # Princess
        # Play Baron (target opp), remaining_card = Princess (9) > Countess (8)
        (0, 3, 1, None, "eliminated_opponent", 9),
    ]
    state = parse_history(history)
    assert 1 in state["eliminated"], state


def test_princess_self_eliminate():
    history = [
        ("r", "facedown", 1),
        ("r", "faceup", ()),
        ("r", "deal0", 9),
        ("r", "deal1", 8),
        ("r", "draw", 0, 2),
        # P0 plays Princess. Remaining = Priest.
        (0, 9, None, None, None, 2),
    ]
    state = parse_history(history)
    assert 0 in state["eliminated"]


def test_priest_reveals_knowledge():
    history = [
        ("r", "facedown", 1),
        ("r", "faceup", ()),
        ("r", "deal0", 2),
        ("r", "deal1", 8),
        ("r", "draw", 0, 1),
        # P0 plays Priest. Knowledge: P1 holds Countess (8). Remaining = Guard.
        (0, 2, 1, None, 8, 1),
    ]
    state = parse_history(history)
    assert state["knowledge"][0] == 8
    assert state["turn"] == 1


def test_countess_forced():
    """If holding King + Countess, must play Countess."""
    history = [
        ("r", "facedown", 1),
        ("r", "faceup", ()),
        ("r", "deal0", 8),   # Countess
        ("r", "deal1", 1),
        ("r", "draw", 0, 7), # King
    ]
    _set_deck(LL.FULL_DECK_COUNTS)
    try:
        actions = LoveLetter().get_available_actions(history)
        decoded = [LL.ACTION_LIST[i] for i in actions]
        assert {a[0] for a in decoded} == {8}, f"should be forced to play Countess, got {decoded}"
    finally:
        _set_deck(_ORIGINAL_DECK)


def test_handmaid_protects():
    history = [
        ("r", "facedown", 1),
        ("r", "faceup", ()),
        ("r", "deal0", 4),
        ("r", "deal1", 8),
        ("r", "draw", 0, 1),
        (0, 4, None, None, None, 1),  # P0 plays Handmaid
    ]
    state = parse_history(history)
    assert 0 in state["handmaid_protected"]
    assert state["turn"] == 1


def test_resolve_effect():
    """_resolve_effect takes an integer target player id."""
    g = LoveLetter()
    state = parse_history([
        ("r", "facedown", 0),
        ("r", "faceup", ()),
        ("r", "deal0", 1),   # P0 Guard
        ("r", "deal1", 2),   # P1 Priest
        ("r", "draw", 0, 4), # P0 drew Handmaid (remaining)
    ])
    # P0 plays Guard targeting opponent (int 1), naming Priest (2). Should eliminate.
    effect = g._resolve_effect(state, player=0, card_played=1, target=1, named_card=2, remaining_card=4)
    assert effect == "eliminated_opponent", f"got {effect}"

    # Baron with higher card vs opponent
    state["hands"][0] = 3
    state["hands"][1] = 2
    effect = g._resolve_effect(state, 0, 3, 1, None, remaining_card=7)  # remaining King > Priest
    assert effect == "eliminated_opponent", f"got {effect}"

    # Prince targeting opponent (not self)
    state["hands"][1] = 9  # Princess
    effect = g._resolve_effect(state, 0, 5, 1, None, remaining_card=4)
    assert effect == "eliminated_opponent", f"got {effect}"

    state["hands"][1] = 4  # Handmaid
    effect = g._resolve_effect(state, 0, 5, 1, None, remaining_card=2)
    assert effect == "opponent_redraws", f"got {effect}"

    # Prince on self
    effect = g._resolve_effect(state, 0, 5, 0, None, remaining_card=9)
    assert effect == "eliminated_self"
    effect = g._resolve_effect(state, 0, 5, 0, None, remaining_card=2)
    assert effect == "self_redraws"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"passed  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
