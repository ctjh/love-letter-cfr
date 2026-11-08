"""
Lazy external-sampling MCCFR+ for Love Letter.

The game tree is never materialised. Each iteration walks a single sampled
trajectory, generating children on demand from the LoveLetter predicates.

  * The traverser alternates between iterations; only their regrets update.
  * Chance nodes sample one outcome from the true distribution.
  * Opponent nodes sample one action from the current strategy.
  * Traverser nodes expand every legal action to compute regrets.
  * CFR+ regret update: regret_sum = max(regret_sum + regret, 0).
  * The average strategy uses linear (iteration-weighted) averaging.
"""

from __future__ import annotations
import pickle
import numpy as np
from tqdm import tqdm

from love_letter import (
    LoveLetter, parse_history, advance_state, ACTION_LIST, ACTION_MAP,
)


# ---------------------------------------------------------------------------
# Per-infoset state
# ---------------------------------------------------------------------------
class InfoSet:
    """Tracks regret and average strategy for one information set.

    Stored sparsely keyed by the legal action indices at this infoset, so the
    arrays are sized to len(legal_actions), NOT the global action count.
    """
    __slots__ = ("legal_actions", "regret_sum", "strategy_sum", "reach_prob_sum")

    def __init__(self, legal_actions):
        n = len(legal_actions)
        self.legal_actions = legal_actions  # np.array of int action indices
        self.regret_sum = np.zeros(n, dtype=np.float64)
        self.strategy_sum = np.zeros(n, dtype=np.float64)
        self.reach_prob_sum = 0.0

    def current_strategy(self) -> np.ndarray:
        """Regret-matching: normalize positive regrets; uniform if all zero."""
        pos = np.maximum(self.regret_sum, 0.0)
        s = pos.sum()
        if s > 0:
            return pos / s
        n = len(self.legal_actions)
        return np.full(n, 1.0 / n)

    def average_strategy(self) -> np.ndarray:
        s = self.strategy_sum.sum()
        if s > 0:
            return self.strategy_sum / s
        n = len(self.legal_actions)
        return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# The recursive traversal
# ---------------------------------------------------------------------------
def cfr(
    game: LoveLetter,
    state: dict,
    infosets: dict,
    traverser: int,
    iteration: int,
    rng: np.random.Generator,
) -> float:
    """Return the sampled value of `state` from `traverser`'s perspective.

    `state` is a parsed game state, advanced one event per recursion with
    advance_state so a traversal is O(depth).

    The two updates happen at different players' nodes:
      * Regrets are updated at the traverser's nodes. Under external sampling the
        returned child values are already counterfactual estimates.
      * The average strategy is accumulated at the opponent's nodes, which are
        reached with exactly the opponent's own reach probability, so adding
        `iteration * strategy` is an unbiased linear-averaging update.
    """
    # ---- terminal -------------------------------------------------------
    if game.is_terminal_node(None, state):
        # get_terminal_utility returns utility from the perspective of the
        # player at the terminal node (game.get_player). Convert to traverser's.
        u_for_actor = game.get_terminal_utility(None, state).get_utility()
        actor = game.get_player(None, state)
        return u_for_actor if actor == traverser else -u_for_actor

    # ---- chance ---------------------------------------------------------
    if game.is_chance_node(None, state):
        outcomes, probs = game.handle_chance(None, state)
        # External sampling: draw ONE outcome from its true distribution.
        idx = rng.choice(len(outcomes), p=probs)
        next_state = advance_state(state, outcomes[idx])
        return cfr(game, next_state, infosets, traverser, iteration, rng)

    # ---- decision -------------------------------------------------------
    legal = game.get_available_actions(None, state)  # np.array of int indices
    key = game.get_infoset_key(None, state)
    infoset = infosets.get(key)
    if infoset is None:
        infoset = InfoSet(legal)
        infosets[key] = infoset

    strategy = infoset.current_strategy()
    player = state["turn"]

    if player == traverser:
        # Branch over all actions and update regret for the traverser.
        utilities = np.zeros(len(legal))
        for i, action_idx in enumerate(legal):
            next_state = _apply_action(game, state, int(action_idx), player)
            utilities[i] = cfr(game, next_state, infosets, traverser, iteration, rng)

        node_utility = float(np.dot(strategy, utilities))
        # CFR+ regret update: clip to zero AFTER adding the new instantaneous regret.
        instantaneous_regret = utilities - node_utility
        infoset.regret_sum = np.maximum(infoset.regret_sum + instantaneous_regret, 0.0)
        return node_utility

    else:
        # Opponent node: accumulate the average strategy, then sample one action.
        infoset.strategy_sum += iteration * strategy
        infoset.reach_prob_sum += iteration
        i = rng.choice(len(legal), p=strategy)
        next_state = _apply_action(game, state, int(legal[i]), player)
        return cfr(game, next_state, infosets, traverser, iteration, rng)


def _apply_action(game: LoveLetter, state: dict, action_idx: int, player: int) -> dict:
    """Resolve `action_idx` for `player` and return the advanced state."""
    card_played, target_token, named_card = ACTION_LIST[action_idx]
    if target_token == "opp":
        target = 1 - player
    elif target_token == "self":
        target = player
    else:
        target = None

    held = state["hands"][player]
    drawn = state["drawn"][player]
    remaining_card = drawn if card_played == held else held

    effect = game._resolve_effect(
        state, player, card_played, target, named_card, remaining_card
    )
    action_event = (player, card_played, target, named_card, effect, remaining_card)
    return advance_state(state, action_event)


# ---------------------------------------------------------------------------
# Trainer entry point
# ---------------------------------------------------------------------------
def train(iterations: int = 1000, seed: int = 0, log_every: int = 100,
          eval_every: int = 0):
    """Run external-sampling MCCFR+ for `iterations` and return infosets + EV.

    If eval_every > 0, report exact exploitability every `eval_every`
    iterations. This walks the whole tree, so only use it on the reduced deck.
    """
    game = LoveLetter()
    infosets: dict[str, InfoSet] = {}
    rng = np.random.default_rng(seed)

    total_utility_p0 = 0.0
    root_state = parse_history([])

    for it in tqdm(range(1, iterations + 1), desc="MCCFR+"):
        traverser = (it - 1) % 2
        u = cfr(
            game,
            state=root_state,
            infosets=infosets,
            traverser=traverser,
            iteration=it,
            rng=rng,
        )
        total_utility_p0 += u if traverser == 0 else -u

        if log_every and it % log_every == 0:
            tqdm.write(
                f"iter {it}: |infosets|={len(infosets)} "
                f"running EV(p0)={total_utility_p0 / it:+.4f}"
            )

        if eval_every and it % eval_every == 0:
            from exploitability import exploitability
            expl = exploitability(game, infosets)
            tqdm.write(f"iter {it}: exploitability={expl:+.5f}  (0 = solved)")

    return infosets, total_utility_p0 / iterations


def save_infosets(infosets: dict, path: str) -> None:
    """Pickle infosets as plain lists: key -> (legal, regret, strategy_sum, reach)."""
    with open(path, "wb") as f:
        pickle.dump({k: (v.legal_actions.tolist(), v.regret_sum.tolist(),
                         v.strategy_sum.tolist(), v.reach_prob_sum)
                     for k, v in infosets.items()}, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="compute exact exploitability every N iters (0 = off; "
                             "reduced deck only)")
    parser.add_argument("--show-top", type=int, default=10,
                        help="print this many of the most-visited infosets")
    parser.add_argument("--out", default="infosets.pkl", help="output pickle")
    args = parser.parse_args()

    import love_letter
    print(f"Deck: {love_letter.DECK_COUNTS}")
    print(f"Global action count: {len(ACTION_LIST)}")
    print(f"Action map: {ACTION_MAP}")
    print()

    infosets, ev = train(iterations=args.iters, seed=args.seed,
                         log_every=args.log_every, eval_every=args.eval_every)

    print(f"\nTotal infosets discovered: {len(infosets)}")
    print(f"Final EV(p0): {ev:+.4f}")

    top = sorted(infosets.items(), key=lambda kv: -kv[1].reach_prob_sum)[: args.show_top]
    if top:
        print(f"\nTop {args.show_top} most-visited infosets:")
    for key, infoset in top:
        avg = infoset.average_strategy()
        moves = [
            f"{ACTION_MAP[a]}={p:.2f}"
            for a, p in zip(infoset.legal_actions, avg)
            if p > 0.01
        ]
        print(f"  reach={infoset.reach_prob_sum:.1f}  {key[:120]}")
        print(f"    -> {', '.join(moves)}")

    save_infosets(infosets, args.out)
    print(f"\nSaved {len(infosets)} infosets -> {args.out}")