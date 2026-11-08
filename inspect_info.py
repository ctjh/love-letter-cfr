"""Load and inspect a trained MCCFR+ infoset pickle."""
import ast
import pickle
import sys
import numpy as np

from love_letter import ACTION_MAP


def load(path):
    """Returns dict: infoset_key (str) -> dict with keys
    legal_actions, regret_sum, strategy_sum, reach_prob_sum, avg_strategy."""
    with open(path, 'rb') as f:
        raw = pickle.load(f)

    out = {}
    for key, (legal, regret, strat_sum, reach) in raw.items():
        legal = np.array(legal)
        strat_sum = np.array(strat_sum)
        total = strat_sum.sum()
        avg = strat_sum / total if total > 0 else np.full(len(legal), 1 / len(legal))
        out[key] = {
            'legal_actions': legal,
            'regret_sum': np.array(regret),
            'strategy_sum': strat_sum,
            'reach_prob_sum': reach,
            'avg_strategy': avg,
        }
    return out


def show(info, key, min_prob=0.01):
    """Pretty-print one infoset's average strategy."""
    rec = info[key]
    moves = [
        f"{ACTION_MAP[a]}={p:.3f}"
        for a, p in zip(rec['legal_actions'], rec['avg_strategy'])
        if p >= min_prob
    ]
    print(f"reach={rec['reach_prob_sum']:.1f}")
    print(f"  {key}")
    print(f"  -> {', '.join(moves)}")


def query(info, hand=None, player=None, min_reach=10):
    """Find infosets matching filters. Returns sorted (most-visited first).

    Works for every LL_KEY mode: all keys are stringified tuples whose first two
    fields are (current_player, hand); the remaining fields differ by mode and are
    ignored here.
    """
    results = []
    for key, rec in info.items():
        if rec['reach_prob_sum'] < min_reach:
            continue
        parsed = ast.literal_eval(key)
        kp, khand = parsed[0], parsed[1]
        if player is not None and kp != player:
            continue
        if hand is not None and tuple(sorted(hand)) != khand:
            continue
        results.append((rec['reach_prob_sum'], key, rec))
    results.sort(reverse=True)
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'infosets.pkl'
    info = load(path)
    print(f"Loaded {len(info):,} infosets from {path}\n")

    # Top 20 most-visited
    print("=== TOP 20 MOST-VISITED INFOSETS ===\n")
    top = sorted(info.items(), key=lambda kv: -kv[1]['reach_prob_sum'])[:20]
    for key, _ in top:
        show(info, key)
        print()

    # Example targeted query: how player 0 plays each starting two-card hand,
    # aggregated by hand (most-visited infoset shown per hand).
    print("\n=== STRATEGY BY HAND (player 0, most-visited infoset per hand) ===\n")
    seen_hands = set()
    for reach, key, _ in query(info, player=0, min_reach=50):
        hand = ast.literal_eval(key)[1]
        if hand in seen_hands:
            continue
        seen_hands.add(hand)
        show(info, key)
        print()