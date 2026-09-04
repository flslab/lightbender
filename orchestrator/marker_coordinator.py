#!/usr/bin/env python3
"""
marker_coordinator.py — IR Marker Grid Designer & Localizer

Designs an N×M grid of blinking IR markers using X distinct IDs for FLS
localization.  Optimizes the ID placement so that observing any K markers
(with their relative grid positions) uniquely determines the FLS's
absolute position on the map.

Usage examples:
    # Design a grid
    python marker_coordinator.py design --rows 20 --cols 20 --ids 8 --min-k 3 \\
        --window 6 --spacing 0.05 --method hybrid --output marker_grid

    # Verify a grid
    python marker_coordinator.py verify --grid marker_grid.json --samples 50000

    # Lookup (simulate localization)
    python marker_coordinator.py lookup --grid marker_grid.json \\
        --obs "0,0,3;2,1,7;1,3,1"

    # Visualize
    python marker_coordinator.py visualize --grid marker_grid.json --output grid.png

    # Serve idempotent short-range tile ON/OFF requests over UDP
    python marker_coordinator.py serve --grid marker_grid.json --port 5558
"""

import numpy as np
import argparse
import csv
import json
import socketserver
import sys
import time
import math
import logging
from marker_techniques import MarkerTechniquesMixin
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any, Set, Callable

logger = logging.getLogger(__name__)

DEFAULT_SHORT_RANGE_WINDOW = 2
DEFAULT_SHORT_RANGE_CELL_SPACING = 0.024
DEFAULT_SHORT_RANGE_MARKER_SIZE = 0.006
DEFAULT_MAIN_MARKER_SIZE = 0.010
DEFAULT_FOCAL_LENGTH = 0.00285
DEFAULT_SENSOR_WIDTH = 0.00384
DEFAULT_SENSOR_HEIGHT = 0.0024
DEFAULT_RESOLUTION_WIDTH = 640
DEFAULT_RESOLUTION_HEIGHT = 400
DEFAULT_USABLE_WIDTH_FRACTION = 0.8
DEFAULT_USABLE_HEIGHT_FRACTION = 0.8
DEFAULT_MIN_MARKER_PX = 1.0
DEFAULT_MIN_BBOX_PX = 30.0
DEFAULT_MARKER_CONTROLLER_PORT = 5558


def marker_window_working_range(
    window_size: int,
    cell_spacing: float,
    marker_size: float,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    sensor_width: float = DEFAULT_SENSOR_WIDTH,
    sensor_height: float = DEFAULT_SENSOR_HEIGHT,
    resolution_width: int = DEFAULT_RESOLUTION_WIDTH,
    resolution_height: int = DEFAULT_RESOLUTION_HEIGHT,
    usable_width_fraction: float = DEFAULT_USABLE_WIDTH_FRACTION,
    usable_height_fraction: float = DEFAULT_USABLE_HEIGHT_FRACTION,
    min_marker_px: float = DEFAULT_MIN_MARKER_PX,
    min_bbox_px: float = DEFAULT_MIN_BBOX_PX,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return the pinhole-camera working range for one square marker window."""
    if (not isinstance(window_size, int) or isinstance(window_size, bool) or
            window_size < 1):
        raise ValueError("Marker window size must be a positive integer")
    positive = {
        'cell_spacing': cell_spacing,
        'marker_size': marker_size,
        'focal_length': focal_length,
        'sensor_width': sensor_width,
        'sensor_height': sensor_height,
        'resolution_width': resolution_width,
        'resolution_height': resolution_height,
        'min_marker_px': min_marker_px,
        'min_bbox_px': min_bbox_px,
    }
    for name, value in positive.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (
        ('usable_width_fraction', usable_width_fraction),
        ('usable_height_fraction', usable_height_fraction),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"{name} must be in (0, 1]")

    bbox_width = (window_size - 1) * cell_spacing + marker_size
    bbox_height = (window_size - 1) * cell_spacing + marker_size
    pitch_width = sensor_width / resolution_width
    pitch_height = sensor_height / resolution_height
    if not math.isclose(pitch_width, pitch_height, rel_tol=1e-9):
        raise ValueError(
            "sensor dimensions and resolution must produce equal pixel pitch"
        )
    pixel_pitch = pitch_width
    min_distance = max(
        bbox_width * focal_length /
        (sensor_width * usable_width_fraction),
        bbox_height * focal_length /
        (sensor_height * usable_height_fraction),
    )
    marker_limit = marker_size * focal_length / (min_marker_px * pixel_pitch)
    bbox_limit = max(bbox_width, bbox_height) * focal_length / (
        min_bbox_px * pixel_pitch
    )
    max_distance = min(marker_limit, bbox_limit)
    if max_distance < min_distance:
        raise ValueError(
            "Marker window has no feasible working range: "
            f"max_distance {max_distance:g} m is below "
            f"min_distance {min_distance:g} m"
        )
    working_range = {
        'min_distance': min_distance,
        'max_distance': max_distance,
    }
    range_model = {
        'model': 'pinhole_marker_window_v1',
        'length_unit': 'm',
        'pixel_pitch_unit': 'm/px',
        'window_rows': window_size,
        'window_cols': window_size,
        'cell_spacing': cell_spacing,
        'marker_size': marker_size,
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'focal_length': focal_length,
        'sensor_width': sensor_width,
        'sensor_height': sensor_height,
        'resolution_width': resolution_width,
        'resolution_height': resolution_height,
        'pixel_pitch_width': pitch_width,
        'pixel_pitch_height': pitch_height,
        'pixel_pitch': pixel_pitch,
        'usable_width_fraction': usable_width_fraction,
        'usable_height_fraction': usable_height_fraction,
        'min_marker_px': min_marker_px,
        'min_bbox_px': min_bbox_px,
        'marker_limit_distance': marker_limit,
        'bbox_limit_distance': bbox_limit,
    }
    return working_range, range_model


# ═══════════════════════════════════════════════════════════════════════════════
#  MARKER GRID
# ═══════════════════════════════════════════════════════════════════════════════

class MarkerGrid(MarkerTechniquesMixin):
    """
    Designs and manages an N×M grid of IR markers using X distinct IDs
    for FLS camera-based localization.

    The grid is constructed so that any K observed markers with their
    relative grid positions uniquely identify the observer's position
    on the map.

    Construction pipeline:
        1. Algebraic seed — polynomial-based ID assignment
        2. Simulated annealing — minimizes shift-agreement conflicts
        3. Verification — sampling-based uniqueness check
    """

    def __init__(self, rows: int, cols: int, num_ids: int, min_k: int,
                 cell_spacing: float = 0.05,
                 grid_origin: Optional[Tuple[float, float, float]] = None,
                 marker_size: float = DEFAULT_MAIN_MARKER_SIZE,
                 window_size: Optional[int] = None):
        """
        Parameters
        ----------
        rows, cols : int
            Grid dimensions.
        num_ids : int
            Number of distinct marker IDs (X).  X << rows*cols.
        min_k : int
            Minimum markers for unique localization.
        cell_spacing : float
            Physical distance between adjacent markers (meters).
        grid_origin : tuple or None
            (x, y, z) world position of cell (0, 0).  When omitted, the
            full grid is centered at the world origin.
        marker_size : float
            Main-grid marker diameter in metres.
        window_size : int or None
            Main-grid marker-window side length.  When omitted, resolve it
            from ``min_k`` using the same rule as grid construction.
        """
        self.rows = rows
        self.cols = cols
        self.num_ids = num_ids
        self.min_k = min_k
        self.cell_spacing = cell_spacing
        self.marker_size = marker_size
        self.window_size = (window_size if window_size is not None else
                            max(2, int(math.ceil(math.sqrt(min_k)))))
        if grid_origin is None:
            grid_origin = (
                (rows - 1) * cell_spacing / 2.0,
                (cols - 1) * cell_spacing / 2.0,
                0.0,
            )
        self.grid_origin = np.array(grid_origin, dtype=float)

        self.grid: Optional[np.ndarray] = None      # rows × cols, dtype int
        self._index: Optional[Dict] = None           # id → list[(r, c)]
        self._agree: Optional[np.ndarray] = None     # (2R-1)×(2C-1) shift agree counts
        self._positions_of: Optional[Dict] = None    # id → set[(r, c)]
        self.short_range: Optional[Dict[str, Any]] = None
        self.working_range: Dict[str, float] = {}
        self.range_model: Dict[str, Any] = {}
        self._range_parameters: Dict[str, Any] = {}
        self.configure_working_ranges()

    def configure_working_ranges(
        self,
        focal_length: float = DEFAULT_FOCAL_LENGTH,
        sensor_width: float = DEFAULT_SENSOR_WIDTH,
        sensor_height: float = DEFAULT_SENSOR_HEIGHT,
        resolution_width: int = DEFAULT_RESOLUTION_WIDTH,
        resolution_height: int = DEFAULT_RESOLUTION_HEIGHT,
        usable_width_fraction: float = DEFAULT_USABLE_WIDTH_FRACTION,
        usable_height_fraction: float = DEFAULT_USABLE_HEIGHT_FRACTION,
        min_marker_px: float = DEFAULT_MIN_MARKER_PX,
        min_bbox_px: float = DEFAULT_MIN_BBOX_PX,
    ) -> None:
        """Store one camera model and apply it to both marker grids."""
        parameters = {
            'focal_length': focal_length,
            'sensor_width': sensor_width,
            'sensor_height': sensor_height,
            'resolution_width': resolution_width,
            'resolution_height': resolution_height,
            'usable_width_fraction': usable_width_fraction,
            'usable_height_fraction': usable_height_fraction,
            'min_marker_px': min_marker_px,
            'min_bbox_px': min_bbox_px,
        }
        working_range, range_model = marker_window_working_range(
            self.window_size, self.cell_spacing, self.marker_size, **parameters
        )
        short_result = None
        if self.short_range is not None:
            short_result = marker_window_working_range(
                self.short_range['window_size'],
                self.short_range['cell_spacing'],
                self.short_range['marker_size'],
                **parameters,
            )
        self._range_parameters = parameters
        self.working_range = working_range
        self.range_model = range_model
        if short_result is not None:
            (self.short_range['working_range'],
             self.short_range['range_model']) = short_result

    def _update_working_ranges(self) -> None:
        self.working_range, self.range_model = marker_window_working_range(
            self.window_size,
            self.cell_spacing,
            self.marker_size,
            **self._range_parameters,
        )
        if self.short_range is not None:
            working_range, range_model = marker_window_working_range(
                self.short_range['window_size'],
                self.short_range['cell_spacing'],
                self.short_range['marker_size'],
                **self._range_parameters,
            )
            self.short_range['working_range'] = working_range
            self.short_range['range_model'] = range_model

    def construct_short_range(
        self,
        window_size: int = DEFAULT_SHORT_RANGE_WINDOW,
        cell_spacing: float = DEFAULT_SHORT_RANGE_CELL_SPACING,
        marker_size: float = DEFAULT_SHORT_RANGE_MARKER_SIZE,
        num_ids: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the dedicated short-range tiles centered in even 2x2 cells.

        Tile ``(i, j)`` is centered on the main-grid window from ``(i, j)``
        through ``(i + 1, j + 1)``.  Its marker IDs start at the main grid's
        ``num_ids``, so every short-range signature is disjoint from every
        main-grid signature.  The smallest ID alphabet that can encode one
        unique row-major signature per tile is used unless ``num_ids`` is
        supplied.
        """
        if self.grid is None:
            raise RuntimeError("Construct the main grid before short-range tiles")
        if window_size < 2:
            raise ValueError("Short-range window size must be at least 2")
        if not math.isfinite(cell_spacing) or cell_spacing <= 0:
            raise ValueError("Short-range cell spacing must be positive and finite")
        if not math.isfinite(marker_size) or marker_size <= 0:
            raise ValueError("Short-range marker size must be positive and finite")

        origins = [
            (i, j)
            for i in range(0, self.rows - 1, 2)
            for j in range(0, self.cols - 1, 2)
        ]
        if not origins:
            raise ValueError("Main grid must contain at least one 2x2 tile")

        signature_length = window_size * window_size
        if num_ids is None:
            num_ids = 1
            while num_ids ** signature_length < len(origins):
                num_ids += 1
        if num_ids < 1:
            raise ValueError("Short-range num_ids must be positive")
        if num_ids ** signature_length < len(origins):
            raise ValueError(
                f"{num_ids} short-range IDs cannot encode {len(origins)} "
                f"unique {window_size}x{window_size} signatures"
            )

        id_offset = self.num_ids
        if id_offset + num_ids > (1 << 16):
            raise ValueError("Combined marker IDs exceed the 16-bit decoder range")

        tiles = []
        for tile_index, (tile_i, tile_j) in enumerate(origins):
            value = tile_index
            digits = [0] * signature_length
            for position in range(signature_length - 1, -1, -1):
                digits[position] = value % num_ids
                value //= num_ids
            signature = [id_offset + digit for digit in digits]

            first = np.asarray(self._cell_to_global(tile_i, tile_j))
            opposite = np.asarray(
                self._cell_to_global(tile_i + 1, tile_j + 1)
            )
            center = (first + opposite) / 2.0
            markers = []
            for local_i in range(window_size):
                for local_j in range(window_size):
                    marker_id = signature[local_i * window_size + local_j]
                    gx = center[0] + ((window_size - 1) / 2.0 - local_i) * cell_spacing
                    gy = center[1] + ((window_size - 1) / 2.0 - local_j) * cell_spacing
                    markers.append({
                        'local_i': local_i,
                        'local_j': local_j,
                        'id': marker_id,
                        'global_x': round(float(gx), 6),
                        'global_y': round(float(gy), 6),
                        'global_z': round(float(center[2]), 6),
                    })

            tiles.append({
                'i': tile_i,
                'j': tile_j,
                'signature': signature,
                'markers': markers,
            })

        self.short_range = {
            'window_size': window_size,
            'cell_spacing': cell_spacing,
            'marker_size': marker_size,
            'id_offset': id_offset,
            'num_ids': num_ids,
            'tiles': tiles,
        }
        self._update_working_ranges()
        return self.short_range

    # ── Construction ─────────────────────────────────────────────────────────

    def construct(self, method: str = 'hybrid', window: Optional[int] = None,
                  seed: int = 42, sa_iterations: int = 200_000,
                  seed_type: str = 'algebraic',
                  verbose: bool = True):
        """
        Build the ID grid.

        Parameters
        ----------
        method : str
            'algebraic'     — fast polynomial seed (may have conflicts)
            'hybrid'        — algebraic seed + simulated annealing (recommended)
            'window_repair' — local collision repair for wxw windows
            'window_repair_optimized'
                            — cached local collision repair for wxw windows
            'de_bruijn'     — de Bruijn torus construction
        window : int or None
            Maximum observation window in grid cells.  If None, inferred
            from grid dimensions.
        seed : int
            Random seed for reproducibility.
        sa_iterations : int
            Number of simulated-annealing steps (hybrid only).
        seed_type : str
            'algebraic' or 'random' initial seed generation type.
        verbose : bool
            Print progress.
        """
        if window is not None:
            self.window_size = window
            self._update_working_ranges()

        rng = np.random.default_rng(seed)
        t0 = time.time()

        if verbose:
            print(f"╔══════════════════════════════════════════════════╗")
            print(f"║  Marker Grid Construction                        ║")
            print(f"╠══════════════════════════════════════════════════╣")
            print(f"║  Grid:  {self.rows}×{self.cols}  "
                  f"({self.rows * self.cols} cells)")
            print(f"║  IDs:   {self.num_ids}    K: {self.min_k}")
            print(f"║  Method: {method}")
            print(f"╚══════════════════════════════════════════════════╝")

        # Phase 1 — initial seed
        if seed_type == 'random':
            self._random_seed(rng, verbose)
        elif seed_type == 'algebraic':
            self._algebraic_seed(rng, verbose)
        else:
            raise ValueError(f"Unknown seed_type: '{seed_type}'")

        # Phase 2 — construction/optimization
        if method == 'hybrid':
            self._init_agree()
            self._init_positions()
            cost_before = self._total_cost()
            if verbose:
                print(f"\n  Initial cost (shift-agreement penalty): {cost_before}")
            self._simulated_annealing(rng, sa_iterations, verbose)
        elif method == 'window_repair':
            w = window if window is not None else max(2, int(math.ceil(math.sqrt(self.min_k))))
            self._window_repair(rng, w, sa_iterations, verbose)
        elif method == 'window_repair_optimized':
            w = window if window is not None else max(2, int(math.ceil(math.sqrt(self.min_k))))
            self._window_repair_optimized(rng, w, sa_iterations, verbose)
        elif method == 'de_bruijn':
            w = window if window is not None else max(2, int(math.ceil(math.sqrt(self.min_k))))
            self._window_repair_deterministic(w, verbose)

        self._build_index()

        if verbose:
            elapsed = time.time() - t0
            print(f"\n  Construction finished in {elapsed:.3f}s")

    # –– Random Seed ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

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

    # ── Window Repair ────────────────────────────────────────────────────────
    
    def _window_repair(self, rng: np.random.Generator, w: int, max_iters: int, verbose: bool):
        """
        Assign IDs to the grid so every w x w window is unique using local collision repair.
        """
        N, M, X = self.rows, self.cols, self.num_ids
        
        if verbose:
            print(f"\n  Phase 2: Window repair (w={w}, max_iters={max_iters})…")

        def get_window(r, c):
            return tuple(self.grid[r + i, c + j] for i in range(w) for j in range(w))

        def get_collisions():
            seen = {}
            conf = []
            for r in range(N - w + 1):
                for c in range(M - w + 1):
                    k = get_window(r, c)
                    if k in seen:
                        conf.append((r, c))
                    else:
                        seen[k] = (r, c)
            return conf

        ok = False
        iters = 0
        for iters in range(max_iters):
            conf = get_collisions()
            if not conf:
                ok = True
                break
            # break a colliding window
            r, c = conf[rng.integers(len(conf))]
            self.grid[r + rng.integers(w), c + rng.integers(w)] = rng.integers(X)

        if verbose:
            if ok:
                print(f"    ✓ Solved in {iters} iterations.")
            else:
                print(f"    ✗ Failed to solve in {max_iters} iterations.")

    def _window_repair_optimized(self, rng: np.random.Generator, w: int,
                                 max_iters: int, verbose: bool):
        """
        Repair duplicate ``w x w`` windows using incrementally updated caches.

        The original implementation rebuilds every window signature after each
        cell mutation. This version caches each signature and the origins at
        which it occurs. A mutation can only affect windows that contain the
        changed cell, so at most ``w**2`` cached entries need to be updated.
        An indexed list of duplicate origins also provides O(1) random choice,
        insertion, and removal.
        """
        N, M, X = self.rows, self.cols, self.num_ids

        if verbose:
            print(f"\n  Phase 2: Optimized window repair "
                  f"(w={w}, max_iters={max_iters})…")

        if w <= 0:
            raise ValueError("Window size must be positive")

        window_rows = N - w + 1
        window_cols = M - w + 1

        # Keep the original method's behavior for grids smaller than a window:
        # there are no window signatures, and therefore no collisions to fix.
        if window_rows <= 0 or window_cols <= 0:
            if verbose:
                print("    ✓ Solved in 0 iterations.")
            return

        window_signatures: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        signature_origins: Dict[
            Tuple[int, ...], Set[Tuple[int, int]]
        ] = defaultdict(set)
        canonical_origin: Dict[Tuple[int, ...], Tuple[int, int]] = {}

        # A list plus reverse index acts as an indexed set. It avoids rebuilding
        # a list from a set on every random selection.
        collisions: List[Tuple[int, int]] = []
        collision_index: Dict[Tuple[int, int], int] = {}

        def add_collision(origin: Tuple[int, int]):
            if origin not in collision_index:
                collision_index[origin] = len(collisions)
                collisions.append(origin)

        def remove_collision(origin: Tuple[int, int]):
            idx = collision_index.pop(origin, None)
            if idx is None:
                return
            last = collisions.pop()
            if idx < len(collisions):
                collisions[idx] = last
                collision_index[last] = idx

        def get_window(origin: Tuple[int, int]) -> Tuple[int, ...]:
            r, c = origin
            return tuple(
                int(self.grid[r + i, c + j])
                for i in range(w) for j in range(w)
            )

        def add_window(origin: Tuple[int, int], signature: Tuple[int, ...]):
            origins = signature_origins[signature]
            if not origins:
                canonical_origin[signature] = origin
            else:
                canonical = canonical_origin[signature]
                if origin < canonical:
                    add_collision(canonical)
                    canonical_origin[signature] = origin
                else:
                    add_collision(origin)
            origins.add(origin)
            window_signatures[origin] = signature

        def remove_window(origin: Tuple[int, int]):
            signature = window_signatures.pop(origin)
            origins = signature_origins[signature]
            canonical = canonical_origin[signature]

            if origin != canonical:
                remove_collision(origin)
                origins.remove(origin)
                return

            origins.remove(origin)
            if not origins:
                del signature_origins[signature]
                del canonical_origin[signature]
                return

            # The previous canonical origin was the only non-collision in this
            # group. Its successor must now be removed from the collision set.
            new_canonical = min(origins)
            canonical_origin[signature] = new_canonical
            remove_collision(new_canonical)

        # Row-major initialization makes the first origin for each signature
        # canonical, matching the collision semantics of _window_repair.
        for r in range(window_rows):
            for c in range(window_cols):
                origin = (r, c)
                add_window(origin, get_window(origin))

        ok = False
        iters = 0
        for iters in range(max_iters):
            if not collisions:
                ok = True
                break

            r, c = collisions[int(rng.integers(len(collisions)))]
            cell_r = r + int(rng.integers(w))
            cell_c = c + int(rng.integers(w))
            new_id = int(rng.integers(X))

            if int(self.grid[cell_r, cell_c]) == new_id:
                continue

            affected = [
                (window_r, window_c)
                for window_r in range(max(0, cell_r - w + 1),
                                      min(cell_r, window_rows - 1) + 1)
                for window_c in range(max(0, cell_c - w + 1),
                                      min(cell_c, window_cols - 1) + 1)
            ]

            for origin in affected:
                remove_window(origin)

            self.grid[cell_r, cell_c] = new_id

            for origin in affected:
                add_window(origin, get_window(origin))

        # The final allowed mutation may have resolved the last collision.
        if not collisions:
            ok = True

        if verbose:
            if ok:
                print(f"    ✓ Solved in {iters} iterations.")
            else:
                print(f"    ✗ Failed to solve in {max_iters} iterations.")

    def _window_repair_deterministic(self, w: int, verbose: bool = False,
                                   max_steps: int = 50_000_000) -> bool:
        """
        Deterministically fill the grid so every w x w window is unique,
        using ordered depth-first backtracking (no randomness).

        Cells are visited in row-major order. Placing a value at (r, c)
        only ever completes a single new window -- the one whose
        bottom-right corner is (r, c), since raster order always finishes
        a window there -- so each placement needs just one O(w^2) check
        against a set of window signatures seen so far, instead of a full
        grid rescan. Candidate ids are tried in ascending order; on a
        dead end (no id works) the search backtracks to the previous cell
        and advances its candidate.

        Returns True if a fully valid grid was found (self.grid is
        updated in place), False if the search exhausted max_steps
        without finding one.
        """
        N, M, X = self.rows, self.cols, self.num_ids
        total = N * M

        if verbose:
            print(f"\n  Phase 2: Deterministic window repair "
                f"(w={w}, backtracking)…")

        seen = set()                  # window signatures placed so far
        next_try = [0] * total        # next candidate id to try at each cell
        added_sig = [None] * total    # signature this cell added, if any

        idx = 0
        steps = 0
        while 0 <= idx < total:
            steps += 1
            if steps > max_steps:
                if verbose:
                    print(f"    ✗ Gave up after {steps:,} steps.")
                return False

            r, c = divmod(idx, M)
            placed = False

            while next_try[idx] < X:
                val = next_try[idx]
                next_try[idx] += 1
                self.grid[r, c] = val

                if r >= w - 1 and c >= w - 1:
                    sig = tuple(
                        int(self.grid[r - w + 1 + i, c - w + 1 + j])
                        for i in range(w) for j in range(w)
                    )
                    if sig in seen:
                        continue          # collision -- try next id
                    seen.add(sig)
                    added_sig[idx] = sig
                else:
                    added_sig[idx] = None

                placed = True
                break

            if placed:
                idx += 1
            else:
                next_try[idx] = 0         # reset for a future visit
                idx -= 1
                if idx < 0:
                    break
                if added_sig[idx] is not None:
                    seen.discard(added_sig[idx])
                    added_sig[idx] = None

        ok = idx == total
        if verbose:
            if ok:
                print(f"    ✓ Solved deterministically in {steps:,} steps.")
            else:
                print(f"    ✗ No valid assignment exists (search exhausted).")
        return ok

    # ── Verification ─────────────────────────────────────────────────────────

    def verify_windows(self, w: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Check uniqueness of all possible contiguous w x w windows in the grid.
        Returns metrics similar to verify().
        """
        N, M = self.rows, self.cols
        
        # Flattened tuple of w*w window -> list of (row, col) occurrences
        windows = {}
        for r in range(N - w + 1):
            for c in range(M - w + 1):
                key = tuple(self.grid[r + i, c + j] for i in range(w) for j in range(w))
                if key not in windows:
                    windows[key] = []
                windows[key].append((r, c))
                
        total_windows = (N - w + 1) * (M - w + 1)
        if total_windows <= 0:
            raise ValueError(f"Grid size {N}x{M} is too small for window size {w}x{w}")
            
        unique = sum(1 for occurrences in windows.values() if len(occurrences) == 1)
        ambiguous = total_windows - unique
        max_matches = max((len(occurrences) for occurrences in windows.values()), default=0)
        
        unique_rate = unique / max(total_windows, 1)
        
        result = {
            'unique_rate': unique_rate,
            'unique_count': unique,
            'ambiguous_count': ambiguous,
            'total_samples': total_windows,
            'max_matches': max_matches,
            'window': w,
        }
        
        if verbose:
            print(f"\n╔══════════════════════════════════════════════════╗")
            print(f"║  Window Verification Report                      ║")
            print(f"╠══════════════════════════════════════════════════╣")
            print(f"║  Window: {w}×{w}    IDs={self.num_ids}")
            print(f"║  Total windows: {total_windows:,}")
            print(f"║  ─────────────────────────────────────────────── ║")
            print(f"║  Unique:    {unique:7,}  ({unique_rate * 100:>5.1f}%)")
            print(f"║  Ambiguous: {ambiguous:7,}  ({(1 - unique_rate) * 100:>5.1f}%)")
            print(f"║  Max matches: {max_matches}")
            print(f"╚══════════════════════════════════════════════════╝\n")
            
        return result

    # ── Lookup ───────────────────────────────────────────────────────────────

    def _build_index(self):
        """Build inverted index: ID → list of (row, col)."""
        self._index = defaultdict(list)
        for r in range(self.rows):
            for c in range(self.cols):
                self._index[int(self.grid[r, c])].append((r, c))

    def lookup(self, observations: List[Tuple[int, int, int]]
               ) -> Optional[List[Dict[str, Any]]]:
        """
        Localize the observer from detected markers.

        Parameters
        ----------
        observations : list of (rel_row, rel_col, marker_id)
            At least K detected markers with relative grid positions.

        Returns
        -------
        - A list of dicts (one per marker) with keys
          {i, j, id, global_x, global_y, global_z} if a unique match is found.
        - A list of such lists if multiple matches are found (ambiguous).
        - None if no match is found.
        """
        if not observations or len(observations) < self.min_k:
            logger.warning(
                f"Need at least {self.min_k} observations, got "
                f"{len(observations)}"
            )
            return None

        if self._index is None:
            self._build_index()

        # Use the rarest ID as anchor for fastest search
        id_counts = {mid: len(self._index.get(mid, []))
                     for _, _, mid in observations}
        anchor_idx = min(range(len(observations)),
                         key=lambda i: id_counts[observations[i][2]])
        dx0, dy0, id0 = observations[anchor_idx]
        other_obs = [o for i, o in enumerate(observations) if i != anchor_idx]

        candidates = self._index.get(id0, [])
        valid: List[Tuple[int, int]] = []

        for (r, c) in candidates:
            R = r - dx0
            C = c - dy0
            match = True
            for dx, dy, mid in other_obs:
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
                valid.append((R, C))

        if len(valid) == 0:
            return None

        results = []
        for R, C in valid:
            markers = []
            for dx, dy, mid in observations:
                gi, gj = R + dx, C + dy
                gx, gy, gz = self._cell_to_global(gi, gj)
                markers.append({
                    'i': gi, 'j': gj, 'id': mid,
                    'global_x': round(gx, 6),
                    'global_y': round(gy, 6),
                    'global_z': round(gz, 6),
                })
            results.append(markers)

        if len(results) == 1:
            return results[0]
        return results  # ambiguous — caller should request more markers

    # ── I/O ──────────────────────────────────────────────────────────────────

    def export_csv(self, filename: str):
        """Export the marker table to CSV."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['i', 'j', 'ID', 'global_x', 'global_y', 'global_z'])
            for i in range(self.rows):
                for j in range(self.cols):
                    gx, gy, gz = self._cell_to_global(i, j)
                    writer.writerow([
                        i, j, int(self.grid[i, j]),
                        f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}"
                    ])
        logger.info(f"Exported CSV → {filename}")

    def export_json(self, filename: str):
        """Export full grid state (grid + metadata) to JSON."""
        self._update_working_ranges()
        data = {
            'rows': self.rows,
            'cols': self.cols,
            'num_ids': self.num_ids,
            'min_k': self.min_k,
            'cell_spacing': self.cell_spacing,
            'marker_size': self.marker_size,
            'window_size': self.window_size,
            'working_range': self.working_range,
            'range_model': self.range_model,
            'grid_origin': self.grid_origin.tolist(),
            'grid': self.grid.tolist(),
            'markers': [],
        }
        for i in range(self.rows):
            for j in range(self.cols):
                gx, gy, gz = self._cell_to_global(i, j)
                data['markers'].append({
                    'i': i, 'j': j, 'id': int(self.grid[i, j]),
                    'global_x': round(gx, 6),
                    'global_y': round(gy, 6),
                    'global_z': round(gz, 6),
                })
        if self.short_range is not None:
            data['short_range'] = self.short_range
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported JSON → {filename}")

    @classmethod
    def from_json(cls, filename: str) -> 'MarkerGrid':
        """Load a MarkerGrid from a JSON file."""
        with open(filename) as f:
            data = json.load(f)
        mg = cls(
            data['rows'], data['cols'], data['num_ids'], data['min_k'],
            data.get('cell_spacing', 0.05),
            tuple(data.get('grid_origin', [0, 0, 0])),
            data.get('marker_size', DEFAULT_MAIN_MARKER_SIZE),
            data.get('window_size'),
        )
        mg.grid = np.array(data['grid'], dtype=int)
        mg.short_range = data.get('short_range')
        range_model = data.get('range_model', {})
        range_keys = tuple(mg._range_parameters)
        if all(key in range_model for key in range_keys):
            mg.configure_working_ranges(**{
                key: range_model[key] for key in range_keys
            })
        else:
            mg._update_working_ranges()
        mg._build_index()
        return mg

    @classmethod
    def from_csv(cls, filename: str, num_ids: Optional[int] = None,
                 min_k: int = 3, cell_spacing: Optional[float] = None,
                 grid_origin: Optional[Tuple[float, float, float]] = None
                 ) -> 'MarkerGrid':
        """Load a grid, inferring omitted geometry from exported coordinates."""
        with open(filename) as f:
            reader = csv.DictReader(f)
            rows_data = list(reader)

        max_i = max(int(r['i']) for r in rows_data)
        max_j = max(int(r['j']) for r in rows_data)
        ids_seen = set(int(r['ID']) for r in rows_data)
        by_cell = {(int(r['i']), int(r['j'])): r for r in rows_data}
        cell_zero = by_cell.get((0, 0))

        global_fields = ('global_x', 'global_y', 'global_z')
        if (grid_origin is None and cell_zero and all(
                cell_zero.get(field) not in (None, '')
                for field in global_fields)):
            grid_origin = tuple(float(cell_zero[field])
                                for field in global_fields)

        if cell_spacing is None:
            for neighbour_cell, field in (((1, 0), 'global_x'),
                                          ((0, 1), 'global_y')):
                neighbour = by_cell.get(neighbour_cell)
                if (cell_zero and neighbour and
                        cell_zero.get(field) not in (None, '') and
                        neighbour.get(field) not in (None, '')):
                    cell_spacing = abs(float(neighbour[field]) -
                                       float(cell_zero[field]))
                    break
            if cell_spacing is None:
                cell_spacing = 0.05

        mg = cls(max_i + 1, max_j + 1,
                 num_ids if num_ids else len(ids_seen),
                 min_k, cell_spacing, grid_origin)
        mg.grid = np.zeros((max_i + 1, max_j + 1), dtype=int)
        for r in rows_data:
            mg.grid[int(r['i']), int(r['j'])] = int(r['ID'])
        mg._build_index()
        return mg

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _cell_to_global(self, i: int, j: int
                        ) -> Tuple[float, float, float]:
        """Convert grid cell (i, j) to world (x, y, z)."""
        gx = self.grid_origin[0] - i * self.cell_spacing
        gy = self.grid_origin[1] - j * self.cell_spacing
        gz = float(self.grid_origin[2])
        return gx, gy, gz

    def print_grid(self, max_display: int = 30):
        """Pretty-print the grid to stdout."""
        if self.grid is None:
            print("Grid not constructed.")
            return
        N = min(self.rows, max_display)
        M = min(self.cols, max_display)
        header = "     " + "".join(f"{j:>3}" for j in range(M))
        print(header)
        print("    " + "─" * (3 * M + 1))
        for i in range(N):
            row = " ".join(f"{self.grid[i, j]:>2}" for j in range(M))
            suffix = " …" if self.cols > max_display else ""
            print(f" {i:>2} │ {row}{suffix}")
        if self.rows > max_display:
            print(f"  …  (showing {N} of {self.rows} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SHORT-RANGE TILE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class MarkerTileController:
    """Idempotent desired-state controller for short-range marker tiles.

    ``apply_tile`` is the hardware-driver contract.  A real deployment supplies
    a callable accepting ``((i, j), enabled)``; the coordinator intentionally
    does not invent a GPIO/SPI mapping that is absent from the grid schema.
    """

    def __init__(self, short_range: Dict[str, Any],
                 apply_tile: Optional[Callable[[Tuple[int, int], bool], None]] = None):
        tiles = short_range.get('tiles', []) if isinstance(short_range, dict) else []
        if not tiles:
            raise ValueError("Marker map has no short-range tiles")

        self._apply_tile = apply_tile
        self._states: Dict[Tuple[int, int], bool] = {}
        for tile in tiles:
            key = (tile.get('i'), tile.get('j'))
            if (not all(isinstance(value, int) and not isinstance(value, bool)
                        for value in key) or min(key) < 0):
                raise ValueError(f"Invalid short-range tile coordinates: {key}")
            if key in self._states:
                raise ValueError(f"Duplicate short-range tile: {key}")
            self._states[key] = True
        if self._apply_tile is not None:
            for key in self._states:
                self._apply_tile(key, True)

    @classmethod
    def from_grid(cls, grid: MarkerGrid,
                  apply_tile: Optional[Callable[[Tuple[int, int], bool], None]] = None
                  ) -> 'MarkerTileController':
        return cls(grid.short_range, apply_tile)

    @property
    def tiles(self) -> Set[Tuple[int, int]]:
        return set(self._states)

    def is_enabled(self, tile: Tuple[int, int]) -> bool:
        if tile not in self._states:
            raise KeyError(f"Unknown short-range tile: {tile}")
        return self._states[tile]

    def set_enabled(self, tile: Tuple[int, int], enabled: bool) -> bool:
        """Apply a desired state and return whether the output changed."""
        if tile not in self._states:
            raise KeyError(f"Unknown short-range tile: {tile}")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        if self._states[tile] == enabled:
            return False
        if self._apply_tile is not None:
            self._apply_tile(tile, enabled)
        self._states[tile] = enabled
        return True

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one decoded UDP request and return its ACK payload."""
        if not isinstance(request, dict):
            raise TypeError("request must be a JSON object")
        request_id = request.get('request_id')
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        raw_tile = request.get('tile')
        if (not isinstance(raw_tile, list) or len(raw_tile) != 2 or
                not all(isinstance(value, int) and not isinstance(value, bool)
                        for value in raw_tile)):
            raise ValueError("tile must contain two integer coordinates")
        enabled = request.get('enabled')
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")

        tile = tuple(raw_tile)
        changed = self.set_enabled(tile, enabled)
        return {
            'request_id': request_id,
            'ok': True,
            'tile': list(tile),
            'enabled': enabled,
            'changed': changed,
        }


class MarkerUDPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        payload, udp_socket = self.request
        request_id = None
        try:
            request = json.loads(payload.decode('utf-8'))
            if isinstance(request, dict):
                request_id = request.get('request_id')
            response = self.server.controller.handle_request(request)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError,
                TypeError, ValueError) as error:
            response = {
                'request_id': request_id,
                'ok': False,
                'error': str(error),
            }
        udp_socket.sendto(
            json.dumps(response, separators=(',', ':')).encode('utf-8'),
            self.client_address,
        )


class MarkerUDPServer(socketserver.UDPServer):
    """Single-threaded UDP server; tile changes are sparse and serialized."""

    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int],
                 controller: MarkerTileController):
        self.controller = controller
        super().__init__(server_address, MarkerUDPRequestHandler)

# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_grid(grid_obj: MarkerGrid, output: Optional[str] = None,
                   show: bool = True):
    """Render the marker grid as a colour-coded plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Circle
    except ImportError:
        print("matplotlib required for visualization.")
        print("Install with:  pip install matplotlib")
        return

    grid = grid_obj.grid
    N, M = grid.shape
    X = grid_obj.num_ids

    # Colour palette
    cmap = plt.cm.get_cmap('tab20', X)

    fig, ax = plt.subplots(1, 1, figsize=(
        min(M * 0.6 + 1, 20),
        min(N * 0.6 + 1, 20),
    ))
    ax.set_xlim(-0.5, M - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Column (j)')
    ax.set_ylabel('Row (i)')
    ax.set_title(f'Marker Grid  {N}×{M}   IDs={X}',
                 fontweight='bold')

    # Draw markers as circles
    for i in range(N):
        for j in range(M):
            mid = int(grid[i, j])
            color = cmap(mid / X)
            circle = Circle((j, i), 0.38, color=color,
                            ec='#333', linewidth=0.5)
            ax.add_patch(circle)
            ax.text(j, i, str(mid), ha='center', va='center',
                    fontsize=max(5, min(10, 200 // max(N, M))),
                    fontweight='bold', color='white' if mid > X // 2 else 'black')

    ax.set_xticks(range(0, M, max(1, M // 10)))
    ax.set_yticks(range(0, N, max(1, N // 10)))
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization → {output}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_design(args):
    """Design a new marker grid."""
    origin = (None if args.origin is None else
              tuple(float(x) for x in args.origin.split(',')))
    window_size = (args.window if args.window is not None else
                   max(2, int(math.ceil(math.sqrt(args.min_k)))))
    mg = MarkerGrid(args.rows, args.cols, args.ids, args.min_k,
                    args.spacing, origin, args.marker_size, window_size)
    mg.configure_working_ranges(
        focal_length=args.focal_length,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        resolution_width=args.resolution_width,
        resolution_height=args.resolution_height,
        usable_width_fraction=args.usable_width_fraction,
        usable_height_fraction=args.usable_height_fraction,
        min_marker_px=args.min_marker_px,
        min_bbox_px=args.min_bbox_px,
    )
    mg.construct(
        method=args.method,
        window=args.window,
        seed=args.seed,
        sa_iterations=args.iterations,
        seed_type=args.seed_type,
        verbose=True,
    )
    if not args.no_short_range:
        short_range = mg.construct_short_range(
            window_size=args.short_window,
            cell_spacing=args.short_spacing,
            marker_size=args.short_marker_size,
            num_ids=args.short_ids,
        )
        print(
            f"\n  Short range: {len(short_range['tiles'])} tiles, "
            f"{short_range['window_size']}x{short_range['window_size']}, "
            f"IDs {short_range['id_offset']}.."
            f"{short_range['id_offset'] + short_range['num_ids'] - 1}"
        )

    mg.print_grid()

    # Export
    json_path = args.output + '.json'
    csv_path = args.output + '.csv'
    mg.export_json(json_path)
    mg.export_csv(csv_path)
    print(f"\n  Exported → {json_path}")
    print(f"  Exported → {csv_path}")

    # Quick verification
    print()
    if args.method in ('window_repair', 'window_repair_optimized', 'de_bruijn'):
        w = args.window if args.window is not None else max(2, int(math.ceil(math.sqrt(mg.min_k))))
        mg.verify_windows(w=w, verbose=True)
    else:
        mg.verify(window=args.window, num_samples=20_000, verbose=True)


def cmd_verify(args):
    """Verify an existing grid."""
    mg = MarkerGrid.from_json(args.grid)
    if args.mode == 'windows':
        w = args.window if args.window is not None else max(2, int(math.ceil(math.sqrt(mg.min_k))))
        mg.verify_windows(w=w, verbose=True)
    else:
        mg.verify(window=args.window, num_samples=args.samples, verbose=True)


def cmd_lookup(args):
    """Lookup position from observations."""
    mg = MarkerGrid.from_json(args.grid)

    # Parse observations: "dx,dy,id;dx,dy,id;..."
    obs = []
    for part in args.obs.split(';'):
        vals = part.strip().split(',')
        if len(vals) != 3:
            print(f"Invalid observation: {part}")
            sys.exit(1)
        obs.append((int(vals[0]), int(vals[1]), int(vals[2])))

    result = mg.lookup(obs)

    if result is None:
        print("  No match found.")
    elif isinstance(result[0], dict):
        # Unique match
        print("\n  ✓ Unique match found!\n")
        print(f"  {'i':>4} {'j':>4} {'ID':>4} {'global_x':>10} "
              f"{'global_y':>10} {'global_z':>10}")
        print(f"  {'─' * 4} {'─' * 4} {'─' * 4} {'─' * 10} "
              f"{'─' * 10} {'─' * 10}")
        for m in result:
            print(f"  {m['i']:>4} {m['j']:>4} {m['id']:>4} "
                  f"{m['global_x']:>10.4f} {m['global_y']:>10.4f} "
                  f"{m['global_z']:>10.4f}")
    else:
        # Ambiguous
        print(f"\n  ⚠ Ambiguous: {len(result)} possible positions.")
        print("  Detect more markers to resolve ambiguity.\n")
        for idx, candidate in enumerate(result):
            print(f"  Candidate {idx + 1}:")
            for m in candidate:
                print(f"    ({m['i']}, {m['j']})  ID={m['id']}  "
                      f"→ ({m['global_x']:.4f}, {m['global_y']:.4f}, "
                      f"{m['global_z']:.4f})")


def cmd_visualize(args):
    """Visualize the grid."""
    mg = MarkerGrid.from_json(args.grid)
    visualize_grid(mg, output=args.output, show=(args.output is None))


def cmd_serve(args):
    """Serve desired-state short-range tile requests over UDP."""
    mg = MarkerGrid.from_json(args.grid)

    def log_driver(tile, enabled):
        state = "ON" if enabled else "OFF"
        print(f"  Tile {tile} -> {state}", flush=True)

    controller = MarkerTileController.from_grid(mg, apply_tile=log_driver)
    with MarkerUDPServer((args.host, args.port), controller) as server:
        host, port = server.server_address
        print(
            f"Marker controller listening on udp://{host}:{port} "
            f"for {len(controller.tiles)} tiles"
        )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            print("\nMarker controller stopped.")


def main():
    parser = argparse.ArgumentParser(
        prog='marker_coordinator',
        description='IR Marker Grid Designer & Localizer for FLS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', help='Available commands')

    # ── design ───────────────────────────────────────────────────────────
    p = sub.add_parser('design', help='Design a new marker grid')
    p.add_argument('--rows', type=int, required=True,
                   help='Grid rows (N)')
    p.add_argument('--cols', type=int, required=True,
                   help='Grid columns (M)')
    p.add_argument('--ids', type=int, required=True,
                   help='Number of distinct IDs (X)')
    p.add_argument('--min-k', type=int, default=3,
                   help='Minimum markers for localization (default: 3)')
    p.add_argument('--window', type=int, default=None,
                   help='Max observation window in grid cells')
    p.add_argument('--spacing', type=float, default=0.05,
                   help='Cell spacing in metres (default: 0.05)')
    p.add_argument('--marker-size', type=float,
                   default=DEFAULT_MAIN_MARKER_SIZE,
                   help='Main marker diameter in metres (default: 0.010)')
    p.add_argument('--focal-length', type=float,
                   default=DEFAULT_FOCAL_LENGTH,
                   help='Camera focal length in metres (default: 0.00285)')
    p.add_argument('--sensor-width', type=float,
                   default=DEFAULT_SENSOR_WIDTH,
                   help='Sensor width in metres (default: 0.00384)')
    p.add_argument('--sensor-height', type=float,
                   default=DEFAULT_SENSOR_HEIGHT,
                   help='Sensor height in metres (default: 0.0024)')
    p.add_argument('--resolution-width', type=int,
                   default=DEFAULT_RESOLUTION_WIDTH,
                   help='Image width in pixels (default: 640)')
    p.add_argument('--resolution-height', type=int,
                   default=DEFAULT_RESOLUTION_HEIGHT,
                   help='Image height in pixels (default: 400)')
    p.add_argument('--usable-width-fraction', type=float,
                   default=DEFAULT_USABLE_WIDTH_FRACTION,
                   help='Usable image-width fraction (default: 0.8)')
    p.add_argument('--usable-height-fraction', type=float,
                   default=DEFAULT_USABLE_HEIGHT_FRACTION,
                   help='Usable image-height fraction (default: 0.8)')
    p.add_argument('--min-marker-px', type=float,
                   default=DEFAULT_MIN_MARKER_PX,
                   help='Minimum detectable marker diameter (default: 1 px)')
    p.add_argument('--min-bbox-px', type=float,
                   default=DEFAULT_MIN_BBOX_PX,
                   help='Minimum separable window bbox (default: 30 px)')
    p.add_argument('--origin', type=str, default=None,
                   help=('World x,y,z of cell (0,0); omit to center the grid '
                         'at world origin'))
    p.add_argument('--method', choices=['algebraic', 'hybrid', 'window_repair',
                                        'window_repair_optimized', 'de_bruijn'],
                   default='hybrid',
                   help='Construction method (default: hybrid)')
    p.add_argument('--seed-type', choices=['algebraic', 'random'],
                   default='algebraic',
                   help='Initial seed generation type (default: algebraic)')
    p.add_argument('--iterations', type=int, default=200_000,
                   help='SA iterations (default: 200000)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default: 42)')
    p.add_argument('--output', type=str, default='marker_grid',
                   help='Output filename without extension')
    p.add_argument('--no-short-range', action='store_true',
                   help='Do not generate the dedicated short-range tiles')
    p.add_argument('--short-window', type=int,
                   default=DEFAULT_SHORT_RANGE_WINDOW,
                   help='Short-range signature side length (default: 2)')
    p.add_argument('--short-spacing', type=float,
                   default=DEFAULT_SHORT_RANGE_CELL_SPACING,
                   help='Short-range marker spacing in metres (default: 0.024)')
    p.add_argument('--short-marker-size', type=float,
                   default=DEFAULT_SHORT_RANGE_MARKER_SIZE,
                   help='Short-range marker diameter in metres (default: 0.006)')
    p.add_argument('--short-ids', type=int, default=None,
                   help='Short-range ID alphabet size (default: minimum needed)')

    # ── verify ───────────────────────────────────────────────────────────
    p = sub.add_parser('verify', help='Verify an existing grid')
    p.add_argument('--grid', type=str, required=True,
                   help='Grid JSON file')
    p.add_argument('--window', type=int, default=None,
                   help='Observation window size')
    p.add_argument('--samples', type=int, default=50_000,
                   help='Number of random tests (default: 50000)')
    p.add_argument('--mode', choices=['subsets', 'windows'], default='subsets',
                   help='Verification mode: subsets (random K-subsets) or windows (all w x w windows)')

    # ── lookup ───────────────────────────────────────────────────────────
    p = sub.add_parser('lookup', help='Lookup position from observations')
    p.add_argument('--grid', type=str, required=True,
                   help='Grid JSON file')
    p.add_argument('--obs', type=str, required=True,
                   help='Observations: "dx,dy,id;dx,dy,id;..."')

    # ── visualize ────────────────────────────────────────────────────────
    p = sub.add_parser('visualize', help='Visualize the grid')
    p.add_argument('--grid', type=str, required=True,
                   help='Grid JSON file')
    p.add_argument('--output', type=str, default=None,
                   help='Output image file (e.g. grid.png)')

    # ── serve ─────────────────────────────────────────────────────────────
    p = sub.add_parser('serve', help='Serve short-range tile ON/OFF requests')
    p.add_argument('--grid', type=str, required=True,
                   help='Grid JSON file containing short_range tiles')
    p.add_argument('--host', type=str, default='0.0.0.0',
                   help='UDP bind address (default: 0.0.0.0)')
    p.add_argument('--port', type=int, default=DEFAULT_MARKER_CONTROLLER_PORT,
                   help='UDP port (default: 5558)')

    args = parser.parse_args()

    if args.command == 'design':
        cmd_design(args)
    elif args.command == 'verify':
        cmd_verify(args)
    elif args.command == 'lookup':
        cmd_lookup(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'serve':
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
