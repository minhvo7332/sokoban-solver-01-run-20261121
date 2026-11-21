// Room-split push-optimal Sokoban solver.
// SPDX-License-Identifier: MIT
//
// Idea: every level is one 1-wide hallway plus 1-4 rooms, each room holding
// 3-4 boxes and as many goals, with a single door onto the hallway. A box
// pushed out of its room can never come back, so boxes stay in their room
// and the level's push-optimal count is the sum of its rooms' optima. We
// therefore solve each room on its own with a push-based A* (state = box
// positions + the player's reachable region, heuristic = cheapest assignment
// of boxes to goals over shortest-path distances, with deadlock pruning)
// and stitch the room plans together with shortest walks through the
// hallway. Push counts, not walk length, decide the score, so the search
// optimises pushes only. All budgets are counts of expanded states, never
// the wall clock, so two runs write byte-identical outputs.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, counted in expanded push-states (never the clock).
const ASTAR_BUDGET = 150_000;
const GREEDY_BUDGET = 150_000;
const GLOBAL_BUDGET = 2_500_000;
let workUsed = 0;

const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goalSet = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch !== '#') wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet[i] = 1;
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// All permutations of 0..n-1, in lexicographic order (deterministic).
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === 1) {
      out.push(a.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      gen(k - 1);
      const j = k % 2 === 1 ? 0 : i;
      const t = a[j]!;
      a[j] = a[k - 1]!;
      a[k - 1] = t;
    }
  };
  gen(n);
  return out;
}

interface Push {
  from: number; // box cell before the push
  dir: number; // index into DIRS
}

interface RegionSpec {
  w: number;
  h: number;
  walk: Uint8Array; // cells the player may stand on (room cells + door entry cell)
  boxOk: Uint8Array; // cells a box may be pushed onto (room cells only)
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  startPlayer: number;
}

// Breadth-first distances from `src` over cells where `pass` is 1.
function bfsDist(
  w: number,
  h: number,
  pass: Uint8Array,
  src: number,
): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  const q: number[] = [src];
  dist[src] = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const x = c % w;
    const y = (c / w) | 0;
    for (const d of DIRS) {
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!pass[n] || dist[n] >= 0) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

// Reachable cells for the player from `start`, avoiding boxes in `boxHere`.
function reachSet(
  w: number,
  h: number,
  walk: Uint8Array,
  boxHere: Uint8Array,
  start: number,
): Uint8Array {
  const seen = new Uint8Array(w * h);
  const q: number[] = [start];
  seen[start] = 1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const x = c % w;
    const y = (c / w) | 0;
    for (const d of DIRS) {
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!walk[n] || boxHere[n] || seen[n]) continue;
      seen[n] = 1;
      q.push(n);
    }
  }
  return seen;
}

function boxHereArray(w: number, h: number, boxes: number[]): Uint8Array {
  const a = new Uint8Array(w * h);
  for (const b of boxes) a[b] = 1;
  return a;
}

// Push-optimal search (A*, then greedy best-first fallback) over one region.
// Returns the push list, or null when the budget runs out unsolved.
function solveRegion(spec: RegionSpec, budget: number): { pushes: Push[]; expansions: number } | null {
  const { w, h, walk, boxOk, goals, goalSet, boxes } = spec;
  const n = boxes.length;
  const perms = permutations(n);
  // Shortest-path distance from each goal over box cells (admissible: one
  // push moves a box one cell, so pushes needed >= this distance).
  const dists = goals.map((g) => bfsDist(w, h, boxOk, g));
  // Dead squares: cells no box can ever be pushed to a goal from, found by
  // reverse pulls from the goals. A box at p can be pushed to q = p + d
  // only if the player can stand at p - d, so in reverse q leads back to
  // p = q - d when q - 2d is walkable. Ignoring other boxes and whether the
  // player can actually reach that stand cell only over-approximates, so
  // every cell left out is truly dead. This catches corners and wall rows
  // with no goal on them.
  const live = new Uint8Array(w * h);
  const liveQ: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    liveQ.push(g);
  }
  for (let qi = 0; qi < liveQ.length; qi++) {
    const q = liveQ[qi]!;
    const qx = q % w;
    const qy = (q / w) | 0;
    for (const d of DIRS) {
      const px = qx - d.dx;
      const py = qy - d.dy;
      const sx = qx - 2 * d.dx;
      const sy = qy - 2 * d.dy;
      if (px < 0 || py < 0 || px >= w || py >= h) continue;
      if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
      const p = py * w + px;
      const s = sy * w + sx;
      if (!boxOk[p] || !walk[s] || live[p]) continue;
      live[p] = 1;
      liveQ.push(p);
    }
  }
  const dead = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (boxOk[i] && !live[i]) dead[i] = 1;

  const hCache = new Map<string, number>();
  const heuristic = (bs: number[]): number => {
    const key = bs.join(',');
    const cached = hCache.get(key);
    if (cached !== undefined) return cached;
    let best = Infinity;
    for (const perm of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = dists[perm[i]!]![bs[i]!]!;
        if (d < 0) {
          s = Infinity;
          break;
        }
        s += d;
        if (s >= best) break;
      }
      if (s < best) best = s;
    }
    hCache.set(key, best);
    return best;
  };

  // 2x2 deadlock: a 2x2 block of walls/boxes holding a box and no goal.
  const isDeadState = (bs: number[], here: Uint8Array): boolean => {
    for (const b of bs) {
      if (dead[b]) return true;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let ox = -1; ox <= 0; ox++) {
        for (let oy = -1; oy <= 0; oy++) {
          const x0 = bx + ox;
          const y0 = by + oy;
          if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
          let blocked = true;
          let hasGoal = false;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const c = (y0 + dy) * w + (x0 + dx);
              if (goalSet[c]) hasGoal = true;
              if (walk[c] && !here[c]) blocked = false;
            }
          }
          if (blocked && !hasGoal) return true;
        }
      }
    }
    return false;
  };

  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b] === 1);

  // Binary heap ordered by (priority, tie-break, insertion counter).
  const heapPri: number[] = [];
  const heapTie: number[] = [];
  const heapCnt: number[] = [];
  const heapIdx: number[] = [];
  let counter = 0;
  const heapLess = (a: number, b: number): boolean =>
    heapPri[a]! < heapPri[b]! ||
    (heapPri[a]! === heapPri[b]! &&
      (heapTie[a]! < heapTie[b]! ||
        (heapTie[a]! === heapTie[b]! && heapCnt[a]! < heapCnt[b]!)));
  const heapPush = (pri: number, tie: number, idx: number): void => {
    let i = heapPri.length;
    heapPri.push(pri);
    heapTie.push(tie);
    heapCnt.push(counter++);
    heapIdx.push(idx);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(i, p)) break;
      for (const arr of [heapPri, heapTie, heapCnt, heapIdx]) {
        const t = arr[i]!;
        arr[i] = arr[p]!;
        arr[p] = t;
      }
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heapIdx[0]!;
    const lastPri = heapPri.pop()!;
    const lastTie = heapTie.pop()!;
    const lastCnt = heapCnt.pop()!;
    const lastIdx = heapIdx.pop()!;
    if (heapPri.length > 0) {
      heapPri[0] = lastPri;
      heapTie[0] = lastTie;
      heapCnt[0] = lastCnt;
      heapIdx[0] = lastIdx;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heapPri.length) break;
        if (c + 1 < heapPri.length && heapLess(c + 1, c)) c++;
        if (!heapLess(c, i)) break;
        for (const arr of [heapPri, heapTie, heapCnt, heapIdx]) {
          const t = arr[i]!;
          arr[i] = arr[c]!;
          arr[c] = t;
        }
        i = c;
      }
    }
    return top;
  };

  const search = (mode: 'astar' | 'greedy', budget: number): { pushes: Push[]; expansions: number } | null => {
    const stBox: number[][] = [boxes.slice().sort((a, b) => a - b)];
    const stPlayer: number[] = [spec.startPlayer];
    const stParent: number[] = [-1];
    const stPushFrom: number[] = [-1];
    const stPushDir: number[] = [-1];
    const stG: number[] = [0];
    const stKey: string[] = [];
    const repOf = (bs: number[], playerPos: number): number => {
      const seen = reachSet(w, h, walk, boxHereArray(w, h, bs), playerPos);
      for (let i = 0; i < w * h; i++) if (seen[i]) return i;
      return playerPos;
    };
    const keyOf = (bs: number[], rep: number): string => `${bs.join(',')}|${rep}`;
    stKey.push(keyOf(stBox[0]!, repOf(stBox[0]!, spec.startPlayer)));
    const best = new Map<string, number>([[stKey[0]!, 0]]);
    const h0 = heuristic(stBox[0]!);
    if (!Number.isFinite(h0)) return null;
    heapPri.length = 0;
    heapTie.length = 0;
    heapCnt.length = 0;
    heapIdx.length = 0;
    counter = 0;
    heapPush(mode === 'astar' ? h0 : h0, h0, 0);
    let expansions = 0;
    while (heapIdx.length > 0) {
      const s = heapPop();
      const g = stG[s]!;
      if (best.get(stKey[s]!) !== g) continue; // a better path to this state exists
      const bs = stBox[s]!;
      if (solved(bs)) {
        const pushes: Push[] = [];
        for (let t = s; t > 0; t = stParent[t]!) {
          pushes.push({ from: stPushFrom[t]!, dir: stPushDir[t]! });
        }
        pushes.reverse();
        return { pushes, expansions };
      }
      if (expansions >= budget || workUsed >= GLOBAL_BUDGET) return null;
      expansions++;
      workUsed++;
      const here = boxHereArray(w, h, bs);
      const reach = reachSet(w, h, walk, here, stPlayer[s]!);
      for (let bi = 0; bi < n; bi++) {
        const b = bs[bi]!;
        const bx = b % w;
        const by = (b / w) | 0;
        for (let d = 0; d < 4; d++) {
          const dir = DIRS[d]!;
          const dx = bx + dir.dx;
          const dy = by + dir.dy;
          if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
          const dest = dy * w + dx;
          if (!boxOk[dest] || here[dest]) continue;
          const behindX = bx - dir.dx;
          const behindY = by - dir.dy;
          if (behindX < 0 || behindY < 0 || behindX >= w || behindY >= h) continue;
          const behind = behindY * w + behindX;
          if (!walk[behind] || !reach[behind]) continue;
          const nb = bs.slice();
          nb[bi] = dest;
          nb.sort((a, b2) => a - b2);
          const nhere = boxHereArray(w, h, nb);
          if (isDeadState(nb, nhere)) continue;
          const nh = heuristic(nb);
          if (!Number.isFinite(nh)) continue;
          const ng = g + 1;
          const rep = repOf(nb, b);
          const k = keyOf(nb, rep);
          const prev = best.get(k);
          if (prev !== undefined && prev <= ng) continue;
          best.set(k, ng);
          const t = stBox.length;
          stBox.push(nb);
          stPlayer.push(b);
          stParent.push(s);
          stPushFrom.push(b);
          stPushDir.push(d);
          stG.push(ng);
          stKey.push(k);
          const pri = mode === 'astar' ? ng + nh : nh;
          const tie = mode === 'astar' ? nh : -ng;
          heapPush(pri, tie, t);
        }
      }
    }
    return null;
  };

  const a = search('astar', budget);
  if (a) return a;
  return search('greedy', GREEDY_BUDGET);
}

// Shortest walk (moves string) from `from` to `to`, treating boxes as walls.
function bfsWalk(
  level: Level,
  boxes: number[],
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const { w, h, wall } = level;
  const blocked = boxHereArray(w, h, boxes);
  const prev = new Int32Array(w * h).fill(-2);
  const prevCh: string[] = new Array(w * h).fill('');
  const q: number[] = [from];
  prev[from] = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const x = c % w;
    const y = (c / w) | 0;
    for (const d of DIRS) {
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (wall[n] || blocked[n] || prev[n] !== -2) continue;
      prev[n] = c;
      prevCh[n] = d.ch;
      if (n === to) {
        let moves = '';
        for (let t = n; t !== from; t = prev[t]!) moves = prevCh[t]! + moves;
        return moves;
      }
      q.push(n);
    }
  }
  return null;
}

// Replay the moves; true only if every step is legal and all boxes end on goals.
function replayOk(level: Level, moves: string): boolean {
  const { w, h, wall, goalSet } = level;
  const boxes = level.boxes.slice();
  let p = level.player;
  const boxAt = (c: number): number => boxes.indexOf(c);
  for (const ch of moves) {
    const d = DIRS.find((dd) => dd.ch === ch);
    if (!d) return false;
    const x = p % w;
    const y = (p / w) | 0;
    const nx = x + d.dx;
    const ny = y + d.dy;
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) return false;
    const n = ny * w + nx;
    if (wall[n]) return false;
    const bi = boxAt(n);
    if (bi >= 0) {
      const bx = nx + d.dx;
      const by = ny + d.dy;
      if (bx < 0 || by < 0 || bx >= w || by >= h) return false;
      const dest = by * w + bx;
      if (wall[dest] || boxAt(dest) >= 0) return false;
      boxes[bi] = dest;
    }
    p = n;
  }
  return boxes.every((b) => goalSet[b] === 1);
}

interface Room {
  cells: number[];
  cellSet: Uint8Array;
  boxes: number[];
  goals: number[];
  door: number;
  entry: number;
  plan: Push[] | null;
}

function solveLevel(level: Level): string {
  const { w, h, wall } = level;
  const hy = (level.player / w) | 0;
  // Split the board into rooms: connected floor components off the hallway row.
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  const boxRoom = new Int32Array(w * h).fill(-1);
  let valid = true;
  for (let start = 0; start < w * h; start++) {
    const sy = (start / w) | 0;
    if (wall[start] || sy === hy || seen[start]) continue;
    const cells: number[] = [start];
    seen[start] = 1;
    for (let qi = 0; qi < cells.length; qi++) {
      const c = cells[qi]!;
      const x = c % w;
      const y = (c / w) | 0;
      for (const d of DIRS) {
        const nx = x + d.dx;
        const ny = y + d.dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || ny === hy || seen[n]) continue;
        seen[n] = 1;
        cells.push(n);
      }
    }
    const cellSet = new Uint8Array(w * h);
    for (const c of cells) cellSet[c] = 1;
    const boxes = cells.filter((c) => level.boxes.includes(c)).sort((a, b) => a - b);
    if (boxes.length === 0) continue;
    const goals = cells.filter((c) => level.goalSet[c] === 1).sort((a, b) => a - b);
    // The door: the room cell next to a hallway floor cell.
    const doors = cells
      .filter((c) => {
        const y = (c / w) | 0;
        const ny = y < hy ? y + 1 : y - 1;
        if (ny !== hy) return false;
        return !wall[ny * w + (c % w)];
      })
      .sort((a, b) => a - b);
    if (doors.length === 0 || goals.length !== boxes.length) {
      valid = false;
      break;
    }
    const door = doors[0]!;
    const dy = (door / w) | 0;
    const entry = (dy < hy ? dy + 1 : dy - 1) * w + (door % w);
    const idx = rooms.length;
    for (const b of boxes) boxRoom[b] = idx;
    rooms.push({ cells, cellSet, boxes, goals, door, entry, plan: null });
  }
  if (valid && level.boxes.some((b) => boxRoom[b] < 0)) valid = false;

  // A room whose solved state leaves a box on its door cell traps the player
  // inside, so such rooms are stitched last. (At most one can exist in a
  // solvable level: after it is finished the player cannot leave it.)
  const solveAll = (rs: Room[]): boolean => {
    for (const r of rs) {
      const walk = new Uint8Array(w * h);
      for (const c of r.cells) walk[c] = 1;
      walk[r.entry] = 1;
      const plan = solveRegion(
        {
          w,
          h,
          walk,
          boxOk: r.cellSet,
          goals: r.goals,
          goalSet: level.goalSet,
          boxes: r.boxes,
          startPlayer: r.entry,
        },
        ASTAR_BUDGET,
      );
      if (!plan) return false;
      r.plan = plan.pushes;
    }
    return true;
  };

  let plans: Room[] | null = null;
  if (valid && rooms.length > 0) {
    if (solveAll(rooms)) plans = rooms;
  }
  if (!plans) {
    // Fallback: one push-optimal search over the whole level.
    const walk = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) if (!wall[i]) walk[i] = 1;
    const plan = solveRegion(
      {
        w,
        h,
        walk,
        boxOk: walk,
        goals: level.goals,
        goalSet: level.goalSet,
        boxes: level.boxes,
        startPlayer: level.player,
      },
      ASTAR_BUDGET,
    );
    if (!plan) return '';
    const whole: Room = {
      cells: [],
      cellSet: walk,
      boxes: level.boxes.slice(),
      goals: level.goals.slice(),
      door: -1,
      entry: level.player,
      plan: plan.pushes,
    };
    plans = [whole];
  }

  // Stitch: order rooms so a door-blocking room (a box ends on its door)
  // comes last; walk the hallway between rooms.
  const finalBoxes = (r: Room): number[] => {
    const bs = r.boxes.slice();
    for (const p of r.plan!) {
      const bi = bs.indexOf(p.from);
      if (bi < 0) return bs;
      const d = DIRS[p.dir]!;
      bs[bi] = p.from + d.dy * w + d.dx;
    }
    return bs;
  };
  const blocksDoor = (r: Room): boolean => r.door >= 0 && finalBoxes(r).includes(r.door);
  const ordered = plans
    .slice()
    .sort((a, b) => (blocksDoor(a) ? 1 : 0) - (blocksDoor(b) ? 1 : 0) || a.entry - b.entry);

  let player = level.player;
  const boxes = level.boxes.slice();
  let moves = '';
  for (const r of ordered) {
    if (r.door >= 0) {
      const walk = bfsWalk(level, boxes, player, r.entry);
      if (walk === null) return '';
      moves += walk;
      player = r.entry;
    }
    for (const p of r.plan!) {
      const d = DIRS[p.dir]!;
      const behind = p.from - d.dy * w - d.dx;
      const walk = bfsWalk(level, boxes, player, behind);
      if (walk === null) return '';
      moves += walk;
      moves += d.ch;
      const bi = boxes.indexOf(p.from);
      if (bi < 0) return '';
      boxes[bi] = p.from + d.dy * w + d.dx;
      player = p.from;
    }
  }
  if (!replayOk(level, moves)) return '';
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8')));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
