"""
Play a round of Love Letter against a trained (near-)equilibrium bot.

The bot is just the saved strategy table from train_full.py / mccfr.py: at each of
its decisions it rebuilds the current information-set key with the SAME code the
solver used (so you must pass the same --key the pickle was trained with), looks up
the average strategy, and samples a move from it. Sampling the mixed equilibrium
(rather than always taking the most-likely move) is what keeps the bot
unexploitable.

A reminder on expectations: a Nash strategy is UNEXPLOITABLE, not unbeatable. Love
Letter has heavy chance, so the bot will lose individual rounds to luck. Its
guarantee is that nobody can beat it in the long run; play a match to several
tokens to feel the edge.

USAGE
  # Play against a trained full-deck bot (use the key it was trained with):
  python play.py --pickle infosets_full_coarse.pkl --key coarse

  # No pickle handy? Train a quick bot on the spot (reduced deck = fast):
  python play.py --deck reduced --key coarse --train 5000

  # Play a match to 3 tokens, you go second:
  python play.py --pickle infosets_full_coarse.pkl --key coarse --tokens 3 --you 1
"""
from __future__ import annotations
import argparse, os, pickle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", default=None, help="trained strategy pickle")
    ap.add_argument("--key", default="coarse", help="infoset key mode the pickle uses")
    ap.add_argument("--deck", default="full", choices=["full", "reduced"])
    ap.add_argument("--train", type=int, default=0,
                    help="if no pickle, train this many MCCFR+ iters first")
    ap.add_argument("--you", type=int, default=0, choices=[0, 1],
                    help="which player you control (default 0 = you move first)")
    ap.add_argument("--tokens", type=int, default=1, help="play a match to this many round wins")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--argmax", action="store_true",
                    help="bot always plays its top move (stronger vs weak play, but exploitable)")
    args = ap.parse_args()

    os.environ["LL_KEY"] = args.key
    os.environ["LL_DECK"] = args.deck

    import numpy as np
    import love_letter
    from love_letter import (
        LoveLetter, parse_history, advance_state, ACTION_LIST, ACTION_MAP, CARD_NAMES,
    )
    from mccfr import _apply_action

    rng = np.random.default_rng(args.seed)

    # ---- load or train the bot strategy -----------------------------------
    strategy = {}  # key -> (legal_actions list, avg_strategy np.array)
    if args.pickle:
        with open(args.pickle, "rb") as f:
            raw = pickle.load(f)
        for k, (legal, _regret, strat_sum, _reach) in raw.items():
            s = np.array(strat_sum, dtype=float)
            avg = s / s.sum() if s.sum() > 0 else np.full(len(legal), 1 / len(legal))
            strategy[k] = (list(legal), avg)
        print(f"Loaded bot: {len(strategy):,} infosets from {args.pickle} (key={args.key})")
    elif args.train:
        from mccfr import train
        print(f"Training a quick bot: {args.train} iters on the {args.deck} deck...")
        infosets, _ = train(iterations=args.train, seed=0, log_every=0)
        for k, inf in infosets.items():
            strategy[k] = (list(inf.legal_actions), inf.average_strategy())
        print(f"Trained bot: {len(strategy):,} infosets")
    else:
        print("No --pickle and no --train: bot will play uniformly at random.")

    game = LoveLetter()
    you, bot = args.you, 1 - args.you
    missing = [0]

    def fmt(card):
        return CARD_NAMES.get(card, str(card))

    def bot_move(state, legal):
        key = game.get_infoset_key(None, state)
        rec = strategy.get(key)
        if rec is None:
            missing[0] += 1
            return int(rng.choice(legal))  # unseen situation -> random fallback
        bot_legal, avg = rec
        # Align avg (indexed by bot_legal) onto the actually-legal actions.
        probs = np.array([avg[bot_legal.index(int(a))] if int(a) in bot_legal else 0.0
                          for a in legal], dtype=float)
        if probs.sum() <= 0:
            return int(rng.choice(legal))
        probs /= probs.sum()
        if args.argmax:
            return int(legal[int(np.argmax(probs))])
        return int(rng.choice(legal, p=probs))

    def human_move(state, legal):
        held, drawn = state["hands"][you], state["drawn"][you]
        print(f"\n  Your hand: {fmt(held)} (held) + {fmt(drawn)} (drawn)")
        if state["knowledge"][you] is not None:
            print(f"  You know the opponent holds: {fmt(state['knowledge'][you])}")
        print("  Your legal moves:")
        for i, a in enumerate(legal):
            print(f"    [{i}] {ACTION_MAP[int(a)]}")
        while True:
            raw = input("  Choose move number: ").strip()
            if raw.isdigit() and 0 <= int(raw) < len(legal):
                return int(legal[int(raw)])
            print("  invalid; try again.")

    def play_round():
        state = parse_history([])
        while True:
            if game.is_terminal_node(None, state):
                return winner(state)
            if game.is_chance_node(None, state):
                outs, ps = game.handle_chance(None, state)
                state = advance_state(state, outs[int(rng.choice(len(outs), p=ps))])
                continue
            legal = game.get_available_actions(None, state)
            mover = state["turn"]
            if mover == you:
                a = human_move(state, legal)
            else:
                a = bot_move(state, legal)
                print(f"\n  Bot plays: {ACTION_MAP[a]}")
            state = _apply_action(game, state, a, mover)

    def winner(state):
        if len(state["eliminated"]) > 0:
            elim = next(iter(state["eliminated"]))
            w = 1 - elim
            print(f"\n  {'You' if elim==you else 'Bot'} eliminated.")
        else:
            h0, h1 = state["hands"][0], state["hands"][1]
            print(f"\n  Showdown — you: {fmt(state['hands'][you])}, bot: {fmt(state['hands'][bot])}")
            w = 0 if h0 > h1 else (1 if h1 > h0 else -1)
            if w == -1:  # tie on card -> discard-sum tiebreak (matches the solver)
                s = {p: sum(state["discard"][p]) for p in (0, 1)}
                w = 0 if s[0] > s[1] else (1 if s[1] > s[0] else -1)
        return w

    # ---- match loop -------------------------------------------------------
    score = {you: 0, bot: 0}
    rnd = 0
    while max(score.values()) < args.tokens:
        rnd += 1
        print(f"\n========== ROUND {rnd}  (you={you}, score you {score[you]} : bot {score[bot]}) ==========")
        w = play_round()
        if w == -1:
            print("  Round tied — no token awarded.")
        else:
            score[w] += 1
            print(f"  Round winner: {'YOU' if w == you else 'BOT'}")
    print(f"\n######### MATCH OVER: {'YOU WIN' if score[you] >= args.tokens else 'BOT WINS'} "
          f"({score[you]} : {score[bot]}) #########")
    if missing[0]:
        print(f"(bot hit {missing[0]} unseen infosets and fell back to random there)")


if __name__ == "__main__":
    main()
