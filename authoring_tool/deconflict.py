import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from logger import setup_logging


matplotlib.use('macosx')

setup_logging()

logger = logging.getLogger(__name__)


# --- Data Structures ---

@dataclass
class Point3D:
    id: int
    x: float
    y: float
    z: float
    length_1: float
    length_2: float
    angle_1: float  # New: Unused in algorithm but stored
    angle_2: float  # New: Unused in algorithm but stored
    max_length_limit: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class InterferenceGraph:
    def __init__(self, points: List[Point3D], threshold: float):
        """
        Constructs the interference graph based on XY plane distance.
        :param points: List of Point3D objects
        :param threshold: Distance threshold for interference
        """
        self.points = {p.id: p for p in points}
        self.threshold = threshold
        self.initial_positions = {p.id: p.position for p in points}
        self.adjacency = self._compute_adjacency()

    def _compute_adjacency(self) -> Dict[int, Set[int]]:
        """Computes adjacency based on XY plane Euclidean distance."""
        adj = {pid: set() for pid in self.points}
        ids = list(self.points.keys())
        n = len(ids)

        for i in range(n):
            for j in range(i + 1, n):
                id_i, id_j = ids[i], ids[j]

                # Use only X and Y for distance calculation
                p1 = self.points[id_i].position[:2]
                p2 = self.points[id_j].position[:2]

                dist = np.linalg.norm(p1 - p2)

                if dist < self.threshold:
                    adj[id_i].add(id_j)
                    adj[id_j].add(id_i)
        return adj

    @property
    def nodes(self) -> List[int]:
        return list(self.points.keys())

    def get_neighbors(self, node_id: int) -> Set[int]:
        return self.adjacency.get(node_id, set())

    def degree(self, node_id: int) -> int:
        return len(self.adjacency.get(node_id, set()))

    def get_z(self, node_id: int) -> float:
        return self.points[node_id].z


# --- Enums for Configuration ---

class SelectionMethod(Enum):
    BRUTE_FORCE = auto()
    GREEDY_MAX_DEGREE = auto()
    GREEDY_TOP_Z = auto()  # Highest Z first
    GREEDY_BOTTOM_Z = auto()  # Lowest Z first
    RANDOM = auto()


class ResolutionOrder(Enum):
    SAME_AS_PHASE_2 = auto()
    MAX_DEGREE = auto()
    TOP_Z = auto()
    BOTTOM_Z = auto()
    RANDOM = auto()


class TrajectoryType(Enum):
    POINT_SPECIFIC = auto()  # Camera -> Point
    GLOBAL_CENTROID = auto()  # Camera -> Centroid


class MoveDirection(Enum):
    AWAY_FROM_CAMERA = auto()
    TOWARDS_CAMERA = auto()


class PlacementType(Enum):
    MIN_DISTANCE = auto()
    LAYERS = auto()


# --- Phase 2: Selection Strategies ---

class SelectionStrategy(ABC):
    @abstractmethod
    def select_points_to_move(self, graph: InterferenceGraph) -> List[int]:
        pass


class BruteForceSelection(SelectionStrategy):
    def select_points_to_move(self, graph: InterferenceGraph) -> List[int]:
        nodes = graph.nodes
        n = len(nodes)

        # Collect all edges
        edges = set()
        for u, neighbors in graph.adjacency.items():
            for v in neighbors:
                if u < v:
                    edges.add((u, v))

        if not edges:
            logger.debug("  Brute Force: No edges found. No moves needed.")
            return []

        # Try size k=1 to n
        for k in range(1, n + 1):
            logger.info(f"  Brute Force: Checking subsets of size {k}")
            for subset in itertools.combinations(nodes, k):
                # Check if subset covers all edges
                covered = True
                subset_set = set(subset)
                for u, v in edges:
                    if u not in subset_set and v not in subset_set:
                        covered = False
                        break
                if covered:
                    logger.info(f"  Brute Force: Found minimal cover: {subset}")
                    return list(subset)
        return nodes


class GreedySelection(SelectionStrategy):
    def __init__(self, heuristic: str):
        self.heuristic = heuristic  # 'degree', 'top_z', 'bottom_z', 'random'

    def select_points_to_move(self, graph: InterferenceGraph) -> List[int]:
        # Work on a copy of the graph (adjacency) to simulate removal
        current_adj = {k: set(v) for k, v in graph.adjacency.items()}
        selected_nodes = []

        logger.info(f"  Starting Greedy Selection (Heuristic: {self.heuristic})")

        while True:
            # Find all edges remaining
            edges_remain = False
            for u in current_adj:
                if current_adj[u]:
                    edges_remain = True
                    break
            if not edges_remain:
                break

            # Pick node based on heuristic
            candidates = [n for n in current_adj if len(current_adj[n]) > 0]
            if not candidates:
                break

            best_node = -1

            if self.heuristic == 'degree':
                # Node with max current degree
                best_node = max(candidates, key=lambda n: len(current_adj[n]))
            elif self.heuristic == 'top_z':
                # Node involved in conflict with highest Z
                best_node = max(candidates, key=lambda n: graph.get_z(n))
            elif self.heuristic == 'bottom_z':
                # Node involved in conflict with lowest Z
                best_node = min(candidates, key=lambda n: graph.get_z(n))
            elif self.heuristic == 'random':
                best_node = random.choice(candidates)

            logger.info(f"    Selected node {best_node} (Candidates: {len(candidates)})")
            selected_nodes.append(best_node)

            # Remove node and edges
            neighbors = current_adj[best_node]
            del current_adj[best_node]
            for neighbor in neighbors:
                if neighbor in current_adj:
                    current_adj[neighbor].discard(best_node)

        return selected_nodes


# --- Phase 3: Resolution Strategies ---

class ResolutionStrategy:
    def __init__(self,
                 order: ResolutionOrder,
                 trajectory: TrajectoryType,
                 direction: MoveDirection,
                 placement: PlacementType,
                 camera_pos: np.ndarray,
                 threshold: float,
                 layer_config: dict = None):
        self.order = order
        self.trajectory = trajectory
        self.direction = direction
        self.placement = placement
        self.camera_pos = camera_pos
        self.threshold = threshold
        self.layer_config = layer_config or {'count': 5, 'spacing': 2.0}

    def resolve(self, graph: InterferenceGraph, points_to_move: List[int]) -> Dict[int, np.ndarray]:
        # 1. Determine Processing Order
        ordered_points = self._sort_points(graph, points_to_move)
        logger.info(f"  Resolution Order: {ordered_points}")

        # 2. Initialize finalized positions with fixed points
        final_positions = {pid: graph.initial_positions[pid]
                           for pid in graph.nodes if pid not in points_to_move}

        # Pre-calculate global vector if needed
        global_vec = None
        if self.trajectory == TrajectoryType.GLOBAL_CENTROID:
            centroid = np.mean([p.position for p in graph.points.values()], axis=0)
            global_vec = centroid - self.camera_pos
            if np.linalg.norm(global_vec) > 0:
                global_vec /= np.linalg.norm(global_vec)
            else:
                global_vec = np.array([0, 0, 1])

        # 3. Process each point
        for pid in ordered_points:
            original_pos = graph.initial_positions[pid]
            point_data = graph.points[pid]

            # Determine Trajectory Vector (normalized)
            if self.trajectory == TrajectoryType.POINT_SPECIFIC:
                vec = original_pos - self.camera_pos
                norm = np.linalg.norm(vec)
                traj_vec = vec / norm if norm > 0 else np.array([0, 0, 1])
            else:
                traj_vec = global_vec

            # Flip if moving towards camera
            if self.direction == MoveDirection.TOWARDS_CAMERA:
                traj_vec = -traj_vec

            # Find Position
            new_pos = self._find_position(pid, point_data, traj_vec, final_positions, graph)
            final_positions[pid] = new_pos

        return final_positions

    def _sort_points(self, graph, points):
        if self.order == ResolutionOrder.SAME_AS_PHASE_2:
            return points  # Assuming list passed in order of selection
        elif self.order == ResolutionOrder.MAX_DEGREE:
            return sorted(points, key=lambda n: graph.degree(n), reverse=True)
        elif self.order == ResolutionOrder.TOP_Z:
            return sorted(points, key=lambda n: graph.get_z(n), reverse=True)
        elif self.order == ResolutionOrder.BOTTOM_Z:
            return sorted(points, key=lambda n: graph.get_z(n))
        elif self.order == ResolutionOrder.RANDOM:
            p_copy = list(points)
            random.shuffle(p_copy)
            return p_copy
        return points

    def _check_perspective_constraint(self, point_data: Point3D, current_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """
        Checks if the new position violates the max length constraint based on perspective scaling.
        L_new = L_original * (dist_new / dist_original)
        """
        dist_old = np.linalg.norm(current_pos - self.camera_pos)
        dist_new = np.linalg.norm(new_pos - self.camera_pos)

        if dist_old == 0: return True

        scale_factor = dist_new / dist_old

        # Check both lengths
        l1_new = point_data.length_1 * scale_factor
        l2_new = point_data.length_2 * scale_factor

        if l1_new > point_data.max_length_limit or l2_new > point_data.max_length_limit:
            logger.debug(
                f"      Perspective Limit Exceeded for Point {point_data.id}: L1={l1_new:.2f}, L2={l2_new:.2f} > Limit={point_data.max_length_limit}")
            return False
        return True

    def _find_position(self, pid, point_data, traj_vec, placed_positions, graph):
        original_pos = graph.initial_positions[pid]

        # Define search steps
        step_size = 0.1
        max_steps = 400  # Safety break

        logger.info(f"    Resolving Point {pid}. Trajectory: {traj_vec}")

        if self.placement == PlacementType.MIN_DISTANCE:
            # Iterative search along vector
            for i in range(max_steps):
                current_shift = i * step_size
                candidate_pos = original_pos + (traj_vec * current_shift)

                # 1. Check Perspective Limit
                if not self._check_perspective_constraint(point_data, original_pos, candidate_pos):
                    if self.direction == MoveDirection.AWAY_FROM_CAMERA:
                        logger.info(f"      Stopping search for Point {pid} due to perspective limit.")
                        break

                        # 2. Check Interference (XY projection check)
                if self._is_valid(candidate_pos, placed_positions, graph, current_id=pid):
                    if i > 0:
                        logger.debug(
                            f"      Found valid position for Point {pid} at step {i} ({current_shift:.2f} units)")
                    return candidate_pos

            logger.info(f"      Failed to find valid position for Point {pid}. Returning original.")
            return original_pos

        elif self.placement == PlacementType.LAYERS:
            # Construct layers
            for d in range(self.layer_config['count']):
                shift = d * self.layer_config['spacing']
                candidate_pos = original_pos + (traj_vec * shift)

                logger.info(f"      Testing Layer {d} (Shift: {shift})")

                if not self._check_perspective_constraint(point_data, original_pos, candidate_pos):
                    continue

                if self._is_valid(candidate_pos, placed_positions, graph, current_id=pid):
                    logger.info(f"      Found valid position for Point {pid} at Layer {d}")
                    return candidate_pos

            logger.info(f"      Failed to find valid position in layers for Point {pid}.")
            return original_pos

    def _is_valid(self, candidate_pos, placed_positions, graph, current_id=None):
        """Checks if candidate_pos interferes with any placed points in XY plane."""
        cand_2d = candidate_pos[:2]  # X, Y only

        for neighbor_id, neighbor_pos in placed_positions.items():
            neighbor_2d = neighbor_pos[:2]
            dist = np.linalg.norm(cand_2d - neighbor_2d)
            if dist < self.threshold:
                # Only log if it's not self-interference (though placed_positions shouldn't contain current_id usually)
                if current_id is not None and current_id != neighbor_id:
                    logger.debug(
                        f"      Interference: Point {current_id} overlaps with {neighbor_id} (Dist: {dist:.2f} < {self.threshold})")
                return False
        return True


# --- Main Solver Class ---

class ModularInterferenceSolver:
    def __init__(self,
                 graph: InterferenceGraph,
                 camera_pos: Tuple[float, float, float]):
        self.graph = graph
        self.camera_pos = np.array(camera_pos)

    def solve(self,
              selection_method: SelectionMethod,
              resolution_order: ResolutionOrder,
              trajectory_type: TrajectoryType,
              move_direction: MoveDirection,
              placement_type: PlacementType,
              layer_config: dict = None) -> Tuple[List[int], Dict[int, np.ndarray]]:

        # Phase 2: Selection
        logger.info(f"\n--- Phase 2: Selection ({selection_method.name}) ---")
        selector = self._get_selector(selection_method)
        points_to_move = selector.select_points_to_move(self.graph)
        logger.info(f"Phase 2 Complete: Selected {len(points_to_move)} points to move: {points_to_move}")

        # Phase 3: Resolution
        logger.info(f"\n--- Phase 3: Resolution ({placement_type.name}, Order: {resolution_order.name}) ---")
        resolver = ResolutionStrategy(
            order=resolution_order,
            trajectory=trajectory_type,
            direction=move_direction,
            placement=placement_type,
            camera_pos=self.camera_pos,
            threshold=self.graph.threshold,
            layer_config=layer_config
        )

        final_positions = resolver.resolve(self.graph, points_to_move)

        return points_to_move, final_positions

    def _get_selector(self, method: SelectionMethod) -> SelectionStrategy:
        if method == SelectionMethod.BRUTE_FORCE:
            return BruteForceSelection()
        elif method == SelectionMethod.GREEDY_MAX_DEGREE:
            return GreedySelection('degree')
        elif method == SelectionMethod.GREEDY_TOP_Z:
            return GreedySelection('top_z')
        elif method == SelectionMethod.GREEDY_BOTTOM_Z:
            return GreedySelection('bottom_z')
        elif method == SelectionMethod.RANDOM:
            return GreedySelection('random')
        else:
            raise ValueError(f"Unknown selection method: {method}")


def visualize_solution(graph: InterferenceGraph,
                       moved_indices: List[int],
                       final_positions: Dict[int, np.ndarray],
                       camera_pos: np.ndarray):
    """
    Visualizes the before and after states of the points in the XY plane.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    except Exception as e:
        logger.info(f"Visualization setup failed (plotting backend might be missing): {e}")
        return

    # Extract data
    all_ids = graph.nodes
    orig_coords = np.array([graph.initial_positions[pid][:2] for pid in all_ids])
    final_coords = np.array([final_positions[pid][:2] for pid in all_ids])
    moved_set = set(moved_indices)

    # --- Plot 1: Before ---
    ax1.set_title(f"Before Resolution ({len(moved_indices)} conflicts)")

    # Draw Camera
    ax1.plot(camera_pos[0], camera_pos[1], 'k*', markersize=15, label='Camera')

    # Draw Points
    for i, pid in enumerate(all_ids):
        color = 'red' if pid in moved_set else 'blue'
        ax1.scatter(orig_coords[i, 0], orig_coords[i, 1], c=color, s=50, zorder=3)
        # Draw threshold radius (0.5 * threshold radius)
        circle = plt.Circle(orig_coords[i], graph.threshold / 2, color=color, fill=False, alpha=0.2)
        ax1.add_patch(circle)
        ax1.text(orig_coords[i, 0] + 0.1, orig_coords[i, 1] + 0.1, str(pid), fontsize=8)

    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Plot 2: After ---
    ax2.set_title("After Resolution")
    ax2.plot(camera_pos[0], camera_pos[1], 'k*', markersize=15, label='Camera')

    for i, pid in enumerate(all_ids):
        color = 'orange' if pid in moved_set else 'blue'
        ax2.scatter(final_coords[i, 0], final_coords[i, 1], c=color, s=50, zorder=3)

        # Draw threshold radius
        circle = plt.Circle(final_coords[i], graph.threshold / 2, color=color, fill=False, alpha=0.2)
        ax2.add_patch(circle)

        # Draw movement vectors
        if pid in moved_set:
            start = graph.initial_positions[pid][:2]
            end = final_positions[pid][:2]
            ax2.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                      head_width=0.2, color='black', alpha=0.5, length_includes_head=True, zorder=2)

    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Setup Data
    points_data = []
    np.random.seed(42)
    for i in range(10):
        points_data.append(Point3D(
            id=i,
            x=np.random.rand() * 10,
            y=np.random.rand() * 10,
            z=np.random.rand() * 5 + 10,
            length_1=1.0,
            length_2=0.5,
            angle_1=np.random.rand() * 360,
            angle_2=np.random.rand() * 360,
            max_length_limit=4.0
        ))

    threshold = 2.0
    camera = (5, 5, 0)

    # 2. Build Graph (Now computes adjacency internally)
    graph = InterferenceGraph(points_data, threshold)

    # 3. Initialize Solver
    solver = ModularInterferenceSolver(graph, camera)

    # 4. Run with Configuration
    logger.info("--- Configuration 1: Greedy Degree + Radial + Away + Min Dist ---")
    moved, positions = solver.solve(
        selection_method=SelectionMethod.GREEDY_MAX_DEGREE,
        resolution_order=ResolutionOrder.MAX_DEGREE,
        trajectory_type=TrajectoryType.POINT_SPECIFIC,
        move_direction=MoveDirection.AWAY_FROM_CAMERA,
        placement_type=PlacementType.MIN_DISTANCE
    )

    # Validation Output
    logger.info(f"Graph stats: {len(graph.nodes)} nodes, threshold {graph.threshold}")
    for pid in moved:
        orig = graph.initial_positions[pid]
        new = positions[pid]
        dist_3d = np.linalg.norm(new - orig)
        logger.info(f"Point {pid} moved {dist_3d:.2f} units to {new}")

    # Visualize results
    visualize_solution(graph, moved, positions, np.array(camera))