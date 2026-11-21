# Room-split push-optimal A* with deadlock tables v1

![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)
![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/minhvo7332/sokoban-solver-01-run-20261121/rank.svg)
![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/minhvo7332/sokoban-solver-01-run-20261121/reward.svg)

A Sokoban solver for [sokoban-solver-01-run-20261121](https://make.fun/competitions/sokoban-solver-01-run-20261121):
one program answers 5 instances in as few pushes as possible.

## Method

Every level is one 1-wide hallway plus 1–4 rooms (3–4 boxes each, one door
each). A box pushed out of its room can never come back, so boxes stay in
their room and the level's push-optimal count P\* is the sum of its rooms'
optima. The solver therefore:

1. splits the board into rooms (connected floor components off the hallway
   row) and finds each room's door;
2. solves each room independently with a push-based A\* — state = box
   positions + the player's reachable region, heuristic = cheapest
   assignment of boxes to goals over shortest-path distances — pruned by
   deadlock detection (reverse-pull dead squares, 2x2 blocks);
3. stitches the room plans together with shortest walks through the
   hallway, solving a room whose final box lands on its door cell last
   (that box traps the player inside, so it must be finished last).

Push counts decide the score, so the search optimises pushes only; walk
length is irrelevant. All budgets are counts of expanded states, never the
wall clock, so two runs write byte-identical outputs.

## Practice results

`grade-local --driver process`, 6 runs, identical each time:

| instance | pushes | P\* | points |
|----------|--------|-----|--------|
| i001 | 13 | 13 | 1,000 |
| i002 | 16 | 16 | 1,000 |
| i003 | 29 | 29 | 1,000 |
| i004 | 47 | 47 | 1,000 |
| i005 | 60 | 60 | 1,000 |

**Practice score: 1,000,000 / 1,000,000** (house baseline plain-astar 1.0.0
scores 200,000 on the same set). Whole set runs in ~0.4 s locally.

## Run

```sh
node main.ts
```

Reads `/input/<id>/level.txt`, writes `/output/<id>/solution.txt`
(`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` override the paths, as in the
kit's starter). Node 22, TypeScript erasable syntax only, stdlib only, no
installs.
