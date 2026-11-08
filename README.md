# Love Letter CFR: Solving an Imperfect-Information Card Game

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-%E2%9C%93-013243)
![Algorithm](https://img.shields.io/badge/algorithm-MCCFR%2B-purple)
![Tests](https://img.shields.io/badge/tests-pytest-green)

Computing approximate **Nash equilibrium strategies** for the two-player card game
[Love Letter](https://en.wikipedia.org/wiki/Love_Letter_(card_game)) using
**Counterfactual Regret Minimization (CFR)**, the family of algorithms behind
superhuman poker agents such as Libratus and Pluribus.

The project implements the full game as an extensive-form game, trains strategies with
**external-sampling Monte Carlo CFR+**, and measures solution quality with
**best-response exploitability**: exactly on a reduced deck, and with a validated
Monte Carlo estimator on the full deck, where the game tree is too large to walk.

**Highlights**

- **Reduced deck solved:** exploitability falls to **0.0017** after 1,000 iterations of
  CFR+ and keeps decreasing toward 0.
- **Imperfect-recall plateau found and fixed:** an infoset key that forgets *when* a card
  was drawn gets stuck at **0.049** exploitability whatever the training budget. Moving
  to a perfect-recall key removes the floor.
- **Full deck:** a Monte Carlo best-response estimator, checked against exact values,
  measures the exploitability of abstracted strategies on a game tree with billions of
  nodes.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Methodology](#methodology)
  - [The Game](#the-game)
  - [Counterfactual Regret Minimization](#counterfactual-regret-minimization)
  - [Information-Set Abstraction](#information-set-abstraction)
  - [Evaluation: Exploitability](#evaluation-exploitability)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Love Letter is a short, bluff-heavy card game. Each player holds one hidden card, draws
a second, and plays one of the two. Cards let you guess, peek at, compare against, swap
with, or force a discard from your opponent's hand. The last player standing, or the one
holding the highest card when the deck runs out, wins the round.

Because each player's card is hidden and chance decides every draw, Love Letter is an
**imperfect-information game**. Optimal play is a *mixed* strategy that has to reason
about the opponent's beliefs, so it cannot be found with minimax search. CFR solves this
class of game: in self-play, each player repeatedly reduces its regret for not having
taken each action, and the *average* strategy converges to a Nash equilibrium.

| Setting | Deck | Game-tree size | Approach | Status |
|---|---|---|---|---|
| Reduced | 6 cards (Guard ×2, Priest, Baron, Countess, Princess) | small enough to enumerate | Exact CFR+ / MCCFR+ with perfect recall | **Solved** (exploitability → 0) |
| Full | 19 cards (Spy ×2, Guard ×6, Priest ×2, Baron ×2, Handmaid ×2, Prince ×2, King, Countess, Princess) | billions of nodes | Abstraction + MCCFR+, evaluated by Monte Carlo best response | Solved up to abstraction |

## Results

### Reduced deck: exact convergence

Full-tree CFR+ with exact best-response exploitability (`vanilla_cfr.py`):

| Iteration | Perfect recall (`pr`, 942 infosets) | Imperfect recall (`exact`, 333 infosets) |
|---:|---:|---:|
| 100 | 0.00654 | 0.05073 |
| 200 | 0.00419 | 0.05063 |
| 400 | 0.00283 | 0.05005 |
| 600 | 0.00228 | 0.04973 |
| 800 | 0.00195 | 0.04954 |
| 1,000 | **0.00174** | **0.04940** |

The perfect-recall strategy approaches a Nash equilibrium. The imperfect-recall key
levels off at about 0.05 regardless of training time. This floor is a modelling error,
not an optimisation problem (see [Key Findings](#key-findings)).

### Full deck: abstraction + MCCFR+

On the full deck, MCCFR+ with the `coarse` key trains at about 2–5 iterations per
second in pure Python. Each Monte Carlo exploitability checkpoint walks the best-response
tree for hundreds of sampled deck orders, which takes tens of minutes per checkpoint.
That is the main cost of a full-deck experiment. To produce the convergence table:

```bash
python train_full.py --key coarse --iters 40000 \
    --eval-at 2000 10000 25000 40000 --eval-dets 400 --save-every 5000
```

**Exploitability** is how much an opponent who knows your strategy exactly could win
from you per round, on average, by playing the best response to it (units: rounds won
minus rounds lost, so ±1 per round). A Nash equilibrium has exploitability 0.

## Methodology

### The Game

`love_letter.py` models a round of two-player Love Letter as an extensive-form game with
chance nodes (shuffle, deal and draws), decision nodes and terminal payoffs.

- **Actions** are `(card, target, named_card)` triples encoded as integer indices, for
  example `Guard->Priest`, `Prince->self` or `Baron->opp`. Legality covers Handmaid
  protection and the forced Countess rule.
- **State** is rebuilt from an event history, or advanced one event at a time with
  `advance_state`. Threading state through the recursion makes each traversal O(depth)
  instead of O(depth²).
- **Payoffs** are +1 for a win and −1 for a loss. Showdown ties are broken by the sum
  of each player's discard pile.

### Counterfactual Regret Minimization

At every information set *I*, CFR tracks the cumulative **counterfactual regret** of
each action *a*: how much better the player would have done by always playing *a* at
*I*, weighted by the probability that the opponent and chance reach *I*. The next
strategy comes from **regret matching**:

$$\sigma^{t+1}(I, a) = \frac{\max(R^t(I,a), 0)}{\sum_{b} \max(R^t(I,b), 0)}$$

The **average strategy** $\bar\sigma^T$ converges to a Nash equilibrium in two-player
zero-sum games. Two solvers are included:

| Solver | File | Per-iteration cost | Use |
|---|---|---|---|
| Vanilla CFR+ | `vanilla_cfr.py` | full tree walk | exact reference on the reduced deck |
| External-sampling MCCFR+ | `mccfr.py` | one sampled trajectory per player | main solver; scales to the full deck |

**MCCFR+** samples chance outcomes and opponent actions, and expands every action at the
traverser's own nodes. It uses CFR+ regret clipping (`R ← max(R + r, 0)`) and linear
averaging. The tree is never stored: children are generated on demand from the game
rules.

### Information-Set Abstraction

The information-set key decides which game histories a player is treated as unable to
tell apart. The true key (**perfect recall**: everything the player has seen, in order)
produces an effectively unbounded number of infosets on the full deck, so each is
visited too rarely for regrets to converge. Following the Libratus/Pluribus approach,
the full deck is solved on a **coarser key** and the resulting strategy is then
evaluated in the real game.

| `LL_KEY` | Information kept | Infosets (reduced deck) | Infosets (full deck) | Recall |
|---|---|---|---|---|
| `pr` | hand + ordered public log + full private observation log | 942 | unbounded | perfect (true game) |
| `exact` | hand + ordered public log + latest reveal | 333 | – | imperfect |
| `counts` | hand + discard multisets + known card + protection | 105 | 2.5M+ | imperfect |
| `belief` | hand + exact multiset of unseen cards | – | ~150k | abstraction |
| `belief2` | hand + set of unseen card types + count | – | ~120k | abstraction |
| `coarse` | hand + game-phase bucket + known card + protection | – | ~8k | abstraction |

`belief` summarises a player's beliefs well: the opponent's card and the remaining draw
pile all come from the multiset of cards the player hasn't seen. `coarse` gives up card
counting but has few enough infosets for each one to be visited many times, so it
converges quickly.

### Evaluation: Exploitability

**Exact best response** (`exploitability.py`, reduced deck). For each player, the code
computes the value of a best response to the trained average strategy, then reports
`exploitability = (BR₀ + BR₁) / 2`. The best response has to commit to **one action per
information set**. A naive version that maximises at every history can see the
opponent's hidden card and overstates exploitability. The best responder always uses
the perfect-recall key, so an abstracted strategy is still measured **in the real
game**.

**Monte Carlo best response** (`mc_exploitability.py`, full deck). Walking the full tree
is infeasible, so the estimator samples *M* random deck orders ("determinizations").
Fixing the deck order leaves only the players' decisions, which is a tiny tree, and
uniform shuffles reproduce the true chance distribution. The best responder still groups
nodes by perfect-recall information set across all samples, so it cannot see future
draws. On the reduced deck, the estimate is checked against the exact value:

| Determinizations *M* | MC estimate | Exact | Error |
|---|---|---|---|
| 100 | 0.383 | 0.218 | +0.165 |
| 400 | 0.241 | 0.218 | +0.023 |
| 1,600 | 0.236 | 0.218 | +0.018 |

*(Strategy: 1,500 iterations of MCCFR+, reduced deck.)* The estimate converges toward the
exact value as *M* grows and errs on the high side, so full-deck figures are upper
bounds.

## Key Findings

Building a correct solver turned up four bugs, each of which stopped convergence on its
own:

1. **Average strategy at the wrong nodes.** In external-sampling MCCFR, the average
   strategy has to be accumulated at the *opponent's* nodes, which sampling reaches with
   the right weight. Accumulating it at the traverser's nodes gave an average strategy
   that never converged.
2. **"Cheating" best response.** The first exploitability metric maximised at each
   history independently. That let the best responder use the opponent's hidden card,
   so every strategy looked badly exploitable. The fix was to make the best response
   respect information sets.
3. **Imperfect recall in the infoset key.** The original key stored the current hand and
   one "last revealed card" slot. It treated *"drew this card this turn"* the same as
   *"held this card since the deal"*, even though those histories imply **different
   beliefs** about the opponent's hand. As a result, two independent CFR
   implementations (this one and openCFR) both plateaued at **≈0.05 exploitability** on
   the reduced deck. A perfect-recall key built from a per-player observation log
   brought exploitability to 0.
4. **Quadratic traversal cost.** Re-parsing the full history at every node made each
   traversal O(depth²). Passing the state down the recursion fixed it.

The third finding is the main lesson of the project. When CFR stops improving at a
non-zero exploitability, the problem can be the **information-set definition** rather
than the learning algorithm.

## Project Structure

```
.
├── love_letter.py         # Game: rules, chance, legal actions, payoffs, infoset keys
├── mccfr.py               # External-sampling MCCFR+ (main solver)
├── vanilla_cfr.py         # Full-tree CFR+ (reference solver, reduced deck)
├── exploitability.py      # Exact infoset-respecting best response
├── mc_exploitability.py   # Monte Carlo best response for the full deck
├── train_full.py          # Full-deck training driver with checkpoints and resume
├── inspect_info.py        # Print a trained strategy as move probabilities
├── play.py                # Play against a trained bot in the terminal
├── test_love_letter.py    # Rule tests
└── requirements.txt
```

## Getting Started

**Requirements:** Python 3.10+

```bash
git clone https://github.com/ctjh/love-letter-cfr.git
cd love-letter-cfr
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the tests:

```bash
pytest
```

The deck and information-set key come from environment variables, which are read when
`love_letter` is imported:

| Variable | Values | Default |
|---|---|---|
| `LL_DECK` | `full`, `reduced` | `full` |
| `LL_KEY` | `pr`, `exact`, `counts`, `belief`, `belief2`, `coarse` | `pr` |

## Usage

### 1. Solve the reduced deck exactly

```bash
# Full-tree CFR+, printing exact exploitability every 10% of iterations
LL_DECK=reduced python vanilla_cfr.py 1000

# Sampled MCCFR+ with exact exploitability checkpoints
LL_DECK=reduced python mccfr.py --iters 20000 --eval-every 2000

# Show the imperfect-recall plateau
LL_DECK=reduced LL_KEY=exact python vanilla_cfr.py 1000
```

### 2. Validate the Monte Carlo estimator

```bash
LL_DECK=reduced python mc_exploitability.py
```

### 3. Train on the full deck

```bash
# Fast abstraction (~8k infosets)
python train_full.py --key coarse --iters 40000 \
    --eval-at 2000 10000 25000 40000 --eval-dets 400 --save-every 5000

# Card-counting abstraction (~150k infosets, run for hours or overnight)
python train_full.py --key belief --iters 400000 \
    --eval-at 50000 150000 300000 400000 --eval-dets 600 --save-every 25000

# Continue an interrupted run
python train_full.py --key coarse --iters 80000 --resume
```

`train_full.py` prints a convergence table of iteration, infoset count, self-play EV and
MC exploitability, and saves the strategy to `infosets_full_<key>.pkl`.

### 4. Inspect the learned strategy

```bash
LL_DECK=full LL_KEY=coarse python inspect_info.py infosets_full_coarse.pkl
```

Example output from a reduced-deck strategy. Holding Guard + Princess, the bot mixes its
Guard guesses:

```
reach=81730.0
  (0, (1, 9), (), (), (('D', 1), ('d', 9)))
  -> Guard->Priest=0.293, Guard->Baron=0.376, Guard->Countess=0.131, Guard->Princess=0.200
```

### 5. Play against the bot

```bash
python play.py --pickle infosets_full_coarse.pkl --key coarse --tokens 3

# No trained strategy yet? Train a quick bot on the reduced deck first:
python play.py --deck reduced --key pr --train 5000
```

The bot samples from its mixed equilibrium strategy. It can't be beaten in expectation,
but because the game is luck-heavy it will still lose individual rounds.

## Future Work

- **Faster traversal.** The solver is pure Python at a few iterations per second on the
  full deck. Most of the time goes into copying state (`Counter`/`dict`/`set`). A
  compact mutable state with undo, or a NumPy/Numba/C++ inner loop, would speed up
  every abstraction.
- **Converge the `belief` abstraction** to measure how much exploitability card
  counting removes compared with `coarse`.
- **Learned abstractions or Deep CFR**, replacing hand-designed keys with a neural
  network that generalises across information sets.
- **Full match play**, solving the multi-round token game rather than single rounds.

## References

1. Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). *Regret Minimization in Games with Incomplete Information.* NeurIPS.
2. Lanctot, M., Waugh, K., Zinkevich, M., & Bowling, M. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games.* NeurIPS.
3. Tammelin, O. (2014). *Solving Large Imperfect Information Games Using CFR+.* arXiv:1407.5042.
4. Brown, N., & Sandholm, T. (2018). *Superhuman AI for heads-up no-limit poker: Libratus beats top professionals.* Science.
5. Brown, N., & Sandholm, T. (2019). *Superhuman AI for multiplayer poker.* Science.
6. Stockham, R. [openCFR](https://github.com/stockhamrexa/OpenCFR), the base `Game` interface used here.
