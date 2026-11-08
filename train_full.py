r"""
Solve full-deck Love Letter with an information-set abstraction and
external-sampling MCCFR+, checkpointing Monte-Carlo exploitability.

The full game tree has billions of nodes and an effectively unbounded number of
perfect-recall infosets, so it is solved up to abstraction: train on a coarser
key (--key), then measure exploitability in the real game with
mc_exploitability.py.

  --key coarse   ~8k infosets    hand + phase + known card + protection
  --key belief2  ~120k infosets  hand + unseen card types + unseen count
  --key belief   ~150k infosets  hand + exact unseen-card composition

Examples:
  python train_full.py --key coarse --iters 40000 --eval-at 2000 10000 25000 40000
  python train_full.py --key belief --iters 400000 --eval-at 50000 150000 400000 \
                       --eval-dets 600 --save-every 25000
  python train_full.py --key coarse --iters 80000 --resume
"""
from __future__ import annotations
import argparse, os, pickle, time


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="coarse",
                    choices=["coarse", "belief2", "belief", "counts", "exact", "pr"],
                    help="information-set abstraction (default: coarse)")
    ap.add_argument("--deck", default="full", choices=["full", "reduced"],
                    help="which deck (default: full)")
    ap.add_argument("--iters", type=int, default=30000,
                    help="total MCCFR+ iterations to run (counting any resumed ones)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-at", type=int, nargs="*",
                    default=[2000, 8000, 20000, 30000],
                    help="iterations at which to measure MC exploitability")
    ap.add_argument("--eval-dets", type=int, default=400,
                    help="determinizations per MC exploitability estimate "
                         "(more = tighter, slower)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint the strategy every N iters (0 = only at end)")
    ap.add_argument("--out", default=None,
                    help="output pickle (default: infosets_full_<key>.pkl)")
    ap.add_argument("--resume", action="store_true",
                    help="load --out first and continue training it")
    return ap.parse_args()


def load(path, InfoSet):
    import numpy as np
    with open(path, "rb") as f:
        raw = pickle.load(f)
    infosets = {}
    for k, (legal, regret, strat, reach) in raw.items():
        inf = InfoSet(np.array(legal))
        inf.regret_sum = np.array(regret, dtype=np.float64)
        inf.strategy_sum = np.array(strat, dtype=np.float64)
        inf.reach_prob_sum = reach
        infosets[k] = inf
    return infosets


def main():
    args = parse_args()
    # love_letter reads these at import time.
    os.environ["LL_KEY"] = args.key
    os.environ["LL_DECK"] = args.deck

    import numpy as np
    from tqdm import tqdm
    import love_letter
    from love_letter import LoveLetter, parse_history
    from mccfr import cfr, InfoSet, save_infosets
    from mc_exploitability import mc_exploitability

    out = args.out or f"infosets_full_{args.key}.pkl"
    game = LoveLetter()
    root_state = parse_history([])
    rng = np.random.default_rng(args.seed)

    if args.resume and os.path.exists(out):
        infosets = load(out, InfoSet)
        print(f"resumed {len(infosets)} infosets from {out}")
    else:
        infosets = {}

    print(f"Deck: {love_letter.DECK_COUNTS}")
    print(f"Key: {args.key}   iters: {args.iters}   eval at: {sorted(set(args.eval_at))}")
    eval_at = set(args.eval_at)

    total_ev_p0 = 0.0
    rows = []
    start = time.time()

    for it in tqdm(range(1, args.iters + 1), desc=f"solve-{args.key}"):
        traverser = (it - 1) % 2
        u = cfr(game, root_state, infosets, traverser, it, rng)
        total_ev_p0 += u if traverser == 0 else -u

        if it in eval_at:
            expl = mc_exploitability(game, infosets, num_dets=args.eval_dets, seed=12345)
            rows.append((it, len(infosets), total_ev_p0 / it, expl))
            tqdm.write(
                f"[{time.time()-start:6.0f}s] iter {it:7d}  |I|={len(infosets):7d}  "
                f"EV(p0)={total_ev_p0/it:+.4f}  MC-exploitability={expl:+.4f}"
            )

        if args.save_every and it % args.save_every == 0:
            save_infosets(infosets, out)

    save_infosets(infosets, out)

    print("\n=== convergence (MC-exploitability should fall toward 0) ===")
    print(f"{'iter':>8} {'|infosets|':>11} {'EV(p0)':>9} {'MC-expl':>9}")
    for it, n, ev, ex in rows:
        print(f"{it:>8} {n:>11} {ev:>+9.4f} {ex:>+9.4f}")
    print(f"\nsaved {len(infosets)} infosets -> {out}")


if __name__ == "__main__":
    main()
