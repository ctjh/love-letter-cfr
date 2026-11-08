"""
Full-tree (non-sampled) CFR+ using the same InfoSet class and infoset keys as
mccfr.py. A reference solver for validating MCCFR+ and the exploitability code;
only tractable on the reduced deck.

  LL_DECK=reduced python vanilla_cfr.py 2000
"""
from __future__ import annotations
import numpy as np

from love_letter import LoveLetter, parse_history, advance_state
from mccfr import InfoSet, _apply_action


def cfr(game, state, infosets, p0, p1, traverser, iteration):
    """Counterfactual value of `state` to `traverser`. p0/p1 are reach probs."""
    if game.is_terminal_node(None, state):
        u_actor = game.get_terminal_utility(None, state).get_utility()
        actor = state["turn"]
        return u_actor if actor == traverser else -u_actor

    if game.is_chance_node(None, state):
        outcomes, probs = game.handle_chance(None, state)
        v = 0.0
        for o, pr in zip(outcomes, probs):
            v += pr * cfr(game, advance_state(state, o), infosets, p0, p1, traverser, iteration)
        return v

    legal = game.get_available_actions(None, state)
    key = game.get_infoset_key(None, state)
    infoset = infosets.get(key)
    if infoset is None:
        infoset = InfoSet(legal)
        infosets[key] = infoset

    strategy = infoset.current_strategy()
    player = state["turn"]

    utilities = np.zeros(len(legal))
    node_util = 0.0
    for i, a in enumerate(legal):
        ns = _apply_action(game, state, int(a), player)
        if player == 0:
            utilities[i] = cfr(game, ns, infosets, p0 * strategy[i], p1, traverser, iteration)
        else:
            utilities[i] = cfr(game, ns, infosets, p0, p1 * strategy[i], traverser, iteration)
        node_util += strategy[i] * utilities[i]

    if player == traverser:
        cf_reach = p1 if traverser == 0 else p0
        own_reach = p0 if traverser == 0 else p1
        regret = cf_reach * (utilities - node_util)
        infoset.regret_sum = np.maximum(infoset.regret_sum + regret, 0.0)
        infoset.strategy_sum += iteration * own_reach * strategy
        infoset.reach_prob_sum += iteration * own_reach

    return node_util


def train(iterations, eval_every=0):
    from exploitability import exploitability
    game = LoveLetter()
    infosets = {}
    root = parse_history([])
    for it in range(1, iterations + 1):
        for traverser in (0, 1):
            cfr(game, root, infosets, 1.0, 1.0, traverser, it)
        if eval_every and it % eval_every == 0:
            print(f"iter {it}: |I|={len(infosets)} exploitability={exploitability(game, infosets):+.5f}")
    return infosets


if __name__ == "__main__":
    import sys
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    train(iters, eval_every=max(1, iters // 10))
