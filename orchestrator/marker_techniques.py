import numpy as np
import math
import time
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any, Set

logger = logging.getLogger(__name__)

class MarkerTechniquesMixin:
    """Mixin containing heuristic techniques for MarkerGrid."""
    def _random_seed(self, rng: np.random.Generator, verbose: bool):
        """
        Generate a fully random initial assignment of IDs.
        """
        if verbose:
            print("\n  Phase 1: Random seed…")
        
        self.grid = rng.integers(0, self.num_ids, size=(self.rows, self.cols))
        
        if verbose:
            score = 0
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr == 0 and dc == 0:
                        continue
                    score += self._count_shift_agreements(self.grid, dr, dc)
            print(f"    Random seed quick-score: {score}")

    # ── Algebraic Seed ───────────────────────────────────────────────────────

    def _algebraic_seed(self, rng: np.random.Generator, verbose: bool):
        """
        Search over random quadratic polynomial coefficients and pick
        the assignment with lowest quick-score (small-shift agreements).
        """
        X = self.num_ids
        N, M = self.rows, self.cols
        I, J = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')

        best_score = float('inf')
        best_grid = None
        trials = 500

        if verbose:
            print(f"\n  Phase 1: Algebraic seed (testing {trials} polynomials)…")

        for _ in range(trials):
            a, b, c, d, e, f = rng.integers(0, X, size=6)
            grid = (a * I * I + b * I * J + c * J * J
                    + d * I + e * J + f) % X

            # Quick score: sum of small-shift agreements
            score = 0
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr == 0 and dc == 0:
                        continue
                    score += self._count_shift_agreements(grid, dr, dc)

            if score < best_score:
                best_score = score
                best_grid = grid.copy()

        self.grid = best_grid.astype(int)
        if verbose:
            print(f"    Best polynomial quick-score: {best_score}")

    # ── Shift Agreement Helpers ──────────────────────────────────────────────

    @staticmethod
    def _count_shift_agreements(grid: np.ndarray, dr: int, dc: int) -> int:
        """Count cells where grid[i,j] == grid[i+dr, j+dc]."""
        N, M = grid.shape
        r1s, r1e = max(0, -dr), min(N, N - dr)
        c1s, c1e = max(0, -dc), min(M, M - dc)
        if r1s >= r1e or c1s >= c1e:
            return 0
        return int(np.sum(
            grid[r1s:r1e, c1s:c1e] == grid[r1s + dr:r1e + dr, c1s + dc:c1e + dc]
        ))

    def _init_agree(self):
        """Build the full (2N-1)×(2M-1) shift-agreement count array."""
        N, M = self.rows, self.cols
        self._agree = np.zeros((2 * N - 1, 2 * M - 1), dtype=int)
        for dr in range(-(N - 1), N):
            for dc in range(-(M - 1), M):
                if dr == 0 and dc == 0:
                    continue
                self._agree[dr + N - 1, dc + M - 1] = \
                    self._count_shift_agreements(self.grid, dr, dc)

    def _init_positions(self):
        """Build id → set-of-positions mapping."""
        self._positions_of = defaultdict(set)
        for r in range(self.rows):
            for c in range(self.cols):
                self._positions_of[int(self.grid[r, c])].add((r, c))

    def _total_cost(self) -> int:
        """
        Sum of max(0, agree[s] - K + 1) over all non-zero shifts.
        We want no more than K-1 agreements at any given shift.
        """
        K = self.min_k
        excess = self._agree - (K - 1)
        return int(np.sum(excess[excess > 0]))

    # ── Simulated Annealing ──────────────────────────────────────────────────

    def _find_conflict_cells(self, rng: np.random.Generator
                             ) -> Optional[Tuple[int, int]]:
        """Pick a cell involved in a high-agreement shift (targeted move).

        Uses a cached list of hot shifts, refreshed every 2000 calls.
        Hot-shift scan is vectorized with numpy for O(1)-ish performance.
        """
        N, M, K = self.rows, self.cols, self.min_k

        # Refresh hot-shift cache periodically (vectorized scan)
        if (not hasattr(self, '_hot_cache') or
                self._hot_cache_age >= 2000 or
                self._hot_cache is None):
            # Find all shifts where agreement >= K
            mask = self._agree >= K
            # Zero out the center (zero shift)
            mask[N - 1, M - 1] = False
            indices = np.argwhere(mask)  # each row is [idx_r, idx_c]

            if len(indices) > 0:
                drs = indices[:, 0] - (N - 1)
                dcs = indices[:, 1] - (M - 1)
                vals = self._agree[indices[:, 0], indices[:, 1]]
                self._hot_cache = list(zip(drs.tolist(), dcs.tolist(),
                                           vals.tolist()))
                # Pre-compute weights
                w = (vals - K + 1).astype(float)
                w /= w.sum()
                self._hot_weights = w
            else:
                self._hot_cache = []
                self._hot_weights = None
            self._hot_cache_age = 0

        self._hot_cache_age += 1

        if not self._hot_cache:
            return None

        # Pick a shift weighted by its excess agreements
        idx = int(rng.choice(len(self._hot_cache), p=self._hot_weights))
        dr, dc, _ = self._hot_cache[idx]

        # Pick a random agreeing cell for this shift
        r1s, r1e = max(0, -dr), min(N, N - dr)
        c1s, c1e = max(0, -dc), min(M, M - dc)
        region1 = self.grid[r1s:r1e, c1s:c1e]
        region2 = self.grid[r1s + dr:r1e + dr, c1s + dc:c1e + dc]
        agree_mask = (region1 == region2)
        positions = np.argwhere(agree_mask)
        if len(positions) == 0:
            return None
        pick = positions[rng.integers(len(positions))]
        return (int(pick[0] + r1s), int(pick[1] + c1s))

    def _smart_new_id(self, r: int, c: int, rng: np.random.Generator) -> int:
        """Pick a new ID that minimises conflicts with close neighbours."""
        X = self.num_ids
        old_id = int(self.grid[r, c])

        # Count neighbour ID occurrences in a small radius
        counts = np.zeros(X, dtype=int)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                ri, ci = r + dr, c + dc
                if 0 <= ri < self.rows and 0 <= ci < self.cols:
                    counts[int(self.grid[ri, ci])] += 1

        # Pick the least-used ID that isn't the current one
        counts[old_id] = 999  # exclude current
        # 50% chance: pick minimum, 50%: pick random for exploration
        if rng.random() < 0.5:
            new_id = int(np.argmin(counts))
        else:
            new_id = int(rng.integers(X))
            if new_id == old_id:
                new_id = (old_id + 1 + int(rng.integers(X - 1))) % X
        return new_id

    def _sa_step(self, rng: np.random.Generator, T: float,
                 targeted_ratio: float = 0.5
                 ) -> Tuple[int, bool]:
        """
        Single SA step.  Returns (cost_delta, accepted).

        With probability `targeted_ratio`, mutates a cell from a
        high-conflict shift; otherwise, picks a random cell.
        """
        K = self.min_k
        N, M, X = self.rows, self.cols, self.num_ids

        # ── Cell selection ───────────────────────────────────────────
        if rng.random() < targeted_ratio:
            target = self._find_conflict_cells(rng)
            if target is not None:
                r, c = target
            else:
                r, c = int(rng.integers(N)), int(rng.integers(M))
        else:
            r, c = int(rng.integers(N)), int(rng.integers(M))

        old_id = int(self.grid[r, c])

        # ── ID selection ─────────────────────────────────────────────
        new_id = self._smart_new_id(r, c, rng)
        if new_id == old_id:
            return 0, False

        # ── Compute delta ────────────────────────────────────────────
        shift_deltas: Dict[Tuple[int, int], int] = {}

        for (r2, c2) in self._positions_of[old_id]:
            if r2 == r and c2 == c:
                continue
            dr, dc = r2 - r, c2 - c
            shift_deltas[(dr, dc)] = shift_deltas.get((dr, dc), 0) - 1
            shift_deltas[(-dr, -dc)] = shift_deltas.get((-dr, -dc), 0) - 1

        for (r2, c2) in self._positions_of[new_id]:
            dr, dc = r2 - r, c2 - c
            shift_deltas[(dr, dc)] = shift_deltas.get((dr, dc), 0) + 1
            shift_deltas[(-dr, -dc)] = shift_deltas.get((-dr, -dc), 0) + 1

        delta = 0
        for (dr, dc), net in shift_deltas.items():
            idx_r = dr + N - 1
            idx_c = dc + M - 1
            old_a = int(self._agree[idx_r, idx_c])
            new_a = old_a + net
            delta += max(0, new_a - K + 1) - max(0, old_a - K + 1)

        # ── Accept / reject ──────────────────────────────────────────
        if delta < 0 or (T > 0 and rng.random() < math.exp(
                -delta / max(T, 1e-12))):
            self.grid[r, c] = new_id
            self._positions_of[old_id].discard((r, c))
            self._positions_of[new_id].add((r, c))
            for (dr, dc), net in shift_deltas.items():
                self._agree[dr + N - 1, dc + M - 1] += net
            return delta, True

        return 0, False

    def _simulated_annealing(self, rng: np.random.Generator,
                             max_iter: int, verbose: bool):
        """
        Minimise the shift-agreement cost using enhanced SA.

        Improvements over basic SA:
          • Targeted mutations — 50% of moves pick a cell from a
            high-agreement shift, breaking the worst conflicts first.
          • Smart ID selection — chooses IDs under-represented among
            the cell's neighbours.
          • Periodic reheating — if stalled for 10% of total
            iterations, temporarily raise T.
        """
        K = self.min_k
        current_cost = self._total_cost()
        best_cost = current_cost
        best_grid = self.grid.copy()

        T = 5.0
        T_min = 0.005
        alpha = 1.0 - 5.0 / max_iter
        accept_count = 0
        improve_count = 0
        stall_counter = 0
        reheat_threshold = max(max_iter // 10, 5000)

        if verbose:
            print(f"\n  Phase 2: Simulated annealing "
                  f"({max_iter:,} iterations, targeted)…")

        for it in range(max_iter):
            delta, accepted = self._sa_step(rng, T)

            if accepted:
                current_cost += delta
                accept_count += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_grid = self.grid.copy()
                    improve_count += 1
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                stall_counter += 1

            T = max(T * alpha, T_min)

            # ── Reheat if stalled ────────────────────────────────────
            if stall_counter >= reheat_threshold and T < 1.0:
                T = min(T * 20, 3.0)
                stall_counter = 0
                if verbose:
                    print(f"    [reheat] T → {T:.3f} at iteration {it + 1:,}")

            # ── Progress ─────────────────────────────────────────────
            if verbose and (it + 1) % (max_iter // 10) == 0:
                pct = 100 * (it + 1) / max_iter
                print(f"    [{pct:5.1f}%]  cost={current_cost:>8,}  "
                      f"best={best_cost:>8,}  T={T:.4f}  "
                      f"accepted={accept_count:,}")

            if best_cost == 0:
                if verbose:
                    print(f"    ✓ Zero cost reached at iteration {it + 1:,}")
                break

        # Restore best
        self.grid = best_grid
        self._init_positions()
        self._init_agree()

        if verbose:
            print(f"    Final best cost: {best_cost:,}  "
                  f"(improvements: {improve_count:,})")
        
        self.grid = best_grid

    def _de_bruijn_torus(self, w: int, verbose: bool):
        """
        Construct a De Bruijn torus using deterministic backtracking (DFS).
        This guarantees an actual de Bruijn torus construction, but can be
        slow for very large grids.
        """
        if verbose:
            print(f"\n  Phase 2: De Bruijn torus construction (DFS, w={w})…")

        N, M, X = self.rows, self.cols, self.num_ids
        self.grid = np.zeros((N, M), dtype=int)
        
        seen_windows = set()
        
        def get_window(r, c):
            return tuple(self.grid[r - w + 1 + i, c - w + 1 + j] 
                         for i in range(w) for j in range(w))

        def solve(r, c):
            if r == N:
                return True
                
            next_r = r if c + 1 < M else r + 1
            next_c = (c + 1) % M
            
            check_window = (r >= w - 1 and c >= w - 1)
            
            for val in range(X):
                self.grid[r, c] = val
                
                win = None
                if check_window:
                    win = get_window(r, c)
                    if win in seen_windows:
                        continue
                    seen_windows.add(win)
                    
                if solve(next_r, next_c):
                    return True
                    
                if check_window and win is not None:
                    seen_windows.remove(win)
                    
            return False

        t0 = time.time()
        success = solve(0, 0)
        
        if verbose:
            elapsed = time.time() - t0
            if success:
                print(f"    ✓ Found valid torus via DFS in {elapsed:.2f}s.")
            else:
                print(f"    ✗ Failed to find a valid torus via DFS in {elapsed:.2f}s.")

    def verify(self, window: Optional[int] = None, num_samples: int = 50_000,
               seed: int = 99, verbose: bool = True) -> Dict[str, Any]:
        """
        Estimate the grid's uniqueness rate by sampling random
        K-subset observations.

        Parameters
        ----------
        window : int or None
            Observation window size (grid cells).  If None, uses full grid.
        num_samples : int
            Number of random observations to test.
        seed : int
            Random seed.
        verbose : bool
            Print report.

        Returns
        -------
        dict with keys:
            unique_rate, ambiguous_count, total_samples,
            avg_matches, max_matches, shift_analysis
        """
        rng = np.random.default_rng(seed)
        K = self.min_k
        N, M, X = self.rows, self.cols, self.num_ids
        W = window if window else min(N, M)

        if self._index is None:
            self._build_index()

        unique = 0
        ambiguous = 0
        no_match = 0
        match_counts = []

        for _ in range(num_samples):
            # Random window origin
            max_r = max(0, N - W)
            max_c = max(0, M - W)
            R = int(rng.integers(0, max_r + 1))
            C = int(rng.integers(0, max_c + 1))

            # Random K-subset within window
            w_r = min(W, N - R)
            w_c = min(W, M - C)
            all_offsets = [(dr, dc) for dr in range(w_r) for dc in range(w_c)]
            if len(all_offsets) < K:
                continue
            chosen = rng.choice(len(all_offsets), size=K, replace=False)
            obs = []
            for idx in chosen:
                dr, dc = all_offsets[idx]
                mid = int(self.grid[R + dr, C + dc])
                obs.append((dr, dc, mid))

            # Count matches
            n_matches = self._count_matches(obs)
            match_counts.append(n_matches)
            if n_matches == 1:
                unique += 1
            elif n_matches == 0:
                no_match += 1
            else:
                ambiguous += 1

        unique_rate = unique / max(len(match_counts), 1)
        avg_matches = np.mean(match_counts) if match_counts else 0
        max_matches = int(np.max(match_counts)) if match_counts else 0

        # Shift analysis — top problematic shifts
        shift_analysis = self._shift_analysis_top(top_n=10)

        result = {
            'unique_rate': unique_rate,
            'unique_count': unique,
            'ambiguous_count': ambiguous,
            'no_match_count': no_match,
            'total_samples': len(match_counts),
            'avg_matches': float(avg_matches),
            'max_matches': max_matches,
            'window': W,
            'shift_analysis': shift_analysis,
        }

        if verbose:
            print(f"\n╔══════════════════════════════════════════════════╗")
            print(f"║  Verification Report                             ║")
            print(f"╠══════════════════════════════════════════════════╣")
            print(f"║  Window: {W}×{W}    K={K}    IDs={X}")
            print(f"║  Samples tested: {len(match_counts):,}")
            print(f"║  ─────────────────────────────────────────────── ║")
            print(f"║  Unique:    {unique:>7,}  "
                  f"({100 * unique_rate:5.1f}%)")
            print(f"║  Ambiguous: {ambiguous:>7,}  "
                  f"({100 * ambiguous / max(len(match_counts), 1):5.1f}%)")
            print(f"║  Avg matches:  {avg_matches:.2f}")
            print(f"║  Max matches:  {max_matches}")
            print(f"╠══════════════════════════════════════════════════╣")
            print(f"║  Top problematic shifts (highest agree counts):  ║")
            for dr, dc, cnt in shift_analysis[:5]:
                print(f"║    Δ({dr:+d}, {dc:+d}): {cnt} agreements")
            print(f"╚══════════════════════════════════════════════════╝")

        return result

    def _count_matches(self, observations: List[Tuple[int, int, int]]) -> int:
        """Count how many grid positions match the observation set."""
        if not observations:
            return 0

        dx0, dy0, id0 = observations[0]
        candidates = self._index.get(id0, [])
        count = 0

        for (r, c) in candidates:
            R = r - dx0
            C = c - dy0
            if R < 0 or C < 0:
                continue
            match = True
            for dx, dy, mid in observations[1:]:
                ri = R + dx
                ci = C + dy
                if (ri < 0 or ri >= self.rows or
                        ci < 0 or ci >= self.cols):
                    match = False
                    break
                if self.grid[ri, ci] != mid:
                    match = False
                    break
            if match:
                count += 1

        return count

    def _shift_analysis_top(self, top_n: int = 10) -> List[Tuple[int, int, int]]:
        """Return the top-N shifts with highest agreement counts."""
        if self._agree is None:
            self._init_agree()

        N, M = self.rows, self.cols
        entries = []
        for dr in range(-(N - 1), N):
            for dc in range(-(M - 1), M):
                if dr == 0 and dc == 0:
                    continue
                cnt = int(self._agree[dr + N - 1, dc + M - 1])
                if cnt > 0:
                    entries.append((dr, dc, cnt))
        entries.sort(key=lambda x: -x[2])
        return entries[:top_n]

# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class CameraMapper:
    """
    Converts image-space marker detections to relative integer grid
    coordinates.  Handles **non-adjacent** markers by detecting the
    underlying grid lattice from pairwise displacement analysis.

    Two modes:
        • known_scale — cell spacing and pixels-per-metre are known
        • lattice     — grid basis vectors detected automatically
    """

    def __init__(self, cell_spacing: float = 0.05):
        self.cell_spacing = cell_spacing

    def detections_to_grid(
        self,
        detections: List[Tuple[float, float, int]],
        mode: str = 'auto',
        pixels_per_metre: Optional[float] = None,
    ) -> List[Tuple[int, int, int]]:
        """
        Map image detections to relative integer grid coordinates.

        Parameters
        ----------
        detections : list of (x, y, marker_id)
            Marker positions in image or physical coordinates.
        mode : str
            'known_scale' — use pixels_per_metre + cell_spacing
            'lattice'     — auto-detect grid vectors
            'auto'        — use known_scale if pixels_per_metre given,
                            else lattice
        pixels_per_metre : float or None
            Conversion factor (required for 'known_scale' mode).
            Can be computed as focal_length / altitude.

        Returns
        -------
        list of (grid_row, grid_col, marker_id)
            Integer grid coordinates relative to the first detected marker.
        """
        if len(detections) < 2:
            return [(0, 0, detections[0][2])] if detections else []

        if mode == 'auto':
            mode = 'known_scale' if pixels_per_metre else 'lattice'

        if mode == 'known_scale':
            return self._from_known_scale(detections, pixels_per_metre)
        else:
            return self._from_lattice(detections)

    # ── Known-Scale Mode ─────────────────────────────────────────────────────

    def _from_known_scale(
        self,
        detections: List[Tuple[float, float, int]],
        pixels_per_metre: float,
    ) -> List[Tuple[int, int, int]]:
        """
        Convert pixel detections to grid coords when scale is known.
        Works naturally for non-adjacent markers — gaps produce
        non-consecutive integer coordinates.
        """
        pts = np.array([(d[0], d[1]) for d in detections])
        ids = [d[2] for d in detections]

        # Convert pixels → metres → grid cells
        metres = pts / pixels_per_metre
        grid_float = metres / self.cell_spacing

        # Translate so minimum is near zero
        grid_float -= grid_float.min(axis=0)

        # Round to nearest integer
        grid_int = np.rint(grid_float).astype(int)

        return [(int(grid_int[i, 1]),   # row ← y
                 int(grid_int[i, 0]),   # col ← x
                 ids[i])
                for i in range(len(ids))]

    # ── Lattice Detection Mode ───────────────────────────────────────────────

    def _from_lattice(
        self,
        detections: List[Tuple[float, float, int]],
    ) -> List[Tuple[int, int, int]]:
        """
        Detect the grid lattice from pairwise displacements and assign
        integer coordinates.  Works for non-adjacent, irregularly
        spaced (in pixel space) marker detections.

        Algorithm:
            1. Compute all pairwise displacement vectors.
            2. Estimate the pixel-per-cell scale by finding the unit
               that makes all displacements integer multiples.
            3. Express each marker as integer grid coordinates.

        Handles non-adjacent markers because a displacement like
        (3·s, 2·s) still reveals s through divisibility analysis.
        Robust to measurement noise via rounding tolerance.
        """
        pts = np.array([(d[0], d[1]) for d in detections])
        ids = [d[2] for d in detections]
        n = len(pts)

        if n < 2:
            return [(0, 0, ids[0])] if ids else []

        # Step 1: collect all pairwise displacements per axis
        dxs = []
        dys = []
        for i in range(n):
            for j in range(i + 1, n):
                dxs.append(abs(pts[j][0] - pts[i][0]))
                dys.append(abs(pts[j][1] - pts[i][1]))

        # Step 2: estimate pixel-per-cell unit for each axis
        def find_unit(values: List[float], max_k: int = 12) -> float:
            """
            Find the unit length u such that each value ≈ k·u for
            some integer k.  Tries u = min_val / k for k = 1..max_k
            and picks the u with lowest total rounding error.

            Filters out noise-level values (< 10% of median) first.
            """
            # Remove near-zero values (noise from same-row/column markers)
            vals = sorted([v for v in values if v > 1e-6])
            if not vals:
                return 1.0

            # Filter out noise: values much smaller than median are
            # likely measurement noise, not real grid displacements
            median_val = np.median(vals)
            noise_threshold = 0.1 * median_val
            vals = [v for v in vals if v > noise_threshold]
            if not vals:
                return 1.0

            best_u = vals[0]
            best_score = float('inf')

            # Try each displacement as a potential base, divided by k
            for base_val in vals:
                for k in range(1, max_k + 1):
                    u = base_val / k
                    if u < 1e-6:
                        continue
                    # Score: sum of fractional parts for all values,
                    # plus a small complexity penalty favouring larger
                    # units (fewer implied grid cells = simpler grid)
                    err = 0.0
                    total_cells = 0.0
                    for v in vals:
                        ratio = v / u
                        err += abs(ratio - round(ratio))
                        total_cells += round(ratio)
                    # A unit that divides everything well but implies
                    # 100s of cells is less plausible than one implying
                    # a few cells.  Scale penalty by number of values
                    # to keep it proportional.
                    complexity = 0.01 * total_cells / len(vals)
                    score = err + complexity
                    if score < best_score:
                        best_score = score
                        best_u = u

            return best_u

        unit_x = find_unit(dxs)
        unit_y = find_unit(dys)

        # If all markers share one axis coordinate, fall back
        non_zero_x = [v for v in dxs if v > 1e-6]
        non_zero_y = [v for v in dys if v > 1e-6]
        if not non_zero_x:
            unit_x = unit_y
        if not non_zero_y:
            unit_y = unit_x

        # Cross-reference: for a square grid, both axes should share
        # the same unit.  If one axis has more data points, prefer
        # the smaller derived unit (the larger one is likely a
        # multiple).  Also verify via joint scoring.
        if non_zero_x and non_zero_y:
            # Check if the smaller unit also explains the other axis
            u_min = min(unit_x, unit_y)
            u_max = max(unit_x, unit_y)
            ratio = u_max / u_min
            # If the larger unit is approximately an integer multiple
            # of the smaller, use the smaller for both axes
            if abs(ratio - round(ratio)) < 0.2 and round(ratio) >= 2:
                unit_x = u_min
                unit_y = u_min

        # Step 3: convert to grid coordinates
        origin = pts[0]
        result = []
        for i in range(n):
            rel = pts[i] - origin
            gj = round(rel[0] / unit_x) if unit_x > 1e-6 else 0  # col
            gi = round(rel[1] / unit_y) if unit_y > 1e-6 else 0  # row
            result.append((gi, gj, ids[i]))

        # Normalise so minimum coords are zero
        min_r = min(r for r, c, _ in result)
        min_c = min(c for r, c, _ in result)
        result = [(r - min_r, c - min_c, mid) for r, c, mid in result]

        return result

