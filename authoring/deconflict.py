import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import logging
import yaml
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

try:
    import networkx as nx

    HAS_NX = True
except ImportError:
    HAS_NX = False

# from logger.logger import setup_logging

matplotlib.use('macosx')
sns.set_theme(style="whitegrid", palette="deep")

# setup_logging()

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
    angle_1: float  # Angle in degrees
    angle_2: float  # Angle in degrees
    yaw: float  # Yaw in degrees (Rotation about +Z)
    max_length_limit: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


def check_downwash(p1: np.ndarray, p2: np.ndarray, threshold: float) -> bool:
    """Checks if two points violate the XY projection (downwash) threshold."""
    return np.linalg.norm(p1[:2] - p2[:2]) < threshold


def check_overlap(p1: np.ndarray, p2: np.ndarray, threshold: float) -> bool:
    """Checks if two points violate the 3D distance (overlap) threshold."""
    return np.linalg.norm(p1 - p2) < threshold


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
        self.total_overlap = 0
        self.total_downwash = 0
        self.edge_types = {}  # Store edge conflict type: 'overlap' or 'downwash'
        self.node_overlaps = {pid: 0 for pid in self.points}
        self.node_downwashes = {pid: 0 for pid in self.points}
        self.adjacency = self._compute_adjacency()

    def _compute_adjacency(self) -> Dict[int, Set[int]]:
        """Computes adjacency based on XY plane Euclidean distance."""
        adj = {pid: set() for pid in self.points}
        ids = list(self.points.keys())
        n = len(ids)

        total_overlap = 0
        total_downwash = 0
        for i in range(n):
            for j in range(i + 1, n):
                id_i, id_j = ids[i], ids[j]

                p1_3d = self.points[id_i].position
                p2_3d = self.points[id_j].position

                is_col = check_overlap(p1_3d, p2_3d, self.threshold)
                is_dw = check_downwash(p1_3d, p2_3d, self.threshold)

                if is_col:
                    adj[id_i].add(id_j)
                    adj[id_j].add(id_i)
                    total_overlap += 1
                    self.edge_types[(id_i, id_j)] = 'overlap'
                    self.edge_types[(id_j, id_i)] = 'overlap'
                    self.node_overlaps[id_i] += 1
                    self.node_overlaps[id_j] += 1
                elif is_dw:
                    adj[id_i].add(id_j)
                    adj[id_j].add(id_i)
                    total_downwash += 1
                    self.edge_types[(id_i, id_j)] = 'downwash'
                    self.edge_types[(id_j, id_i)] = 'downwash'
                    self.node_downwashes[id_i] += 1
                    self.node_downwashes[id_j] += 1

        self.total_overlap = total_overlap
        self.total_downwash = total_downwash
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


# --- Helper Functions for Perspective Logic ---

def get_optical_axis(points: List[Point3D], camera_pos: np.ndarray) -> np.ndarray:
    """Computes the optical axis by connecting camera to points centroid."""
    if not points:
        return np.array([0, 0, 1])
    positions = np.array([p.position for p in points])
    centroid = np.mean(positions, axis=0)
    axis = centroid - camera_pos
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.array([0, 0, 1])
    return axis / norm


def calculate_perspective_scale(orig_pos: np.ndarray, new_pos: np.ndarray, camera_pos: np.ndarray,
                                optical_axis: np.ndarray) -> float:
    """
    Calculates scale factor based on distance projected along the optical axis.
    Scale = Dist_New / Dist_Old (where Dist is projection onto optical axis).
    """
    # Project vector from camera to point onto the optical axis
    dist_old = np.dot(orig_pos - camera_pos, optical_axis)
    dist_new = np.dot(new_pos - camera_pos, optical_axis)

    if dist_old <= 1e-9:
        return 1.0  # Avoid division by zero

    return dist_new / dist_old


def calculate_scaled_lengths(point: Point3D, new_pos: np.ndarray, camera_pos: np.ndarray, optical_axis: np.ndarray) -> \
        Tuple[float, float]:
    """Computes the physical lengths of lines after perspective scaling using optical axis distance."""
    orig_pos = np.array([point.x, point.y, point.z])
    scale = calculate_perspective_scale(orig_pos, new_pos, camera_pos, optical_axis)
    return point.length_1 * scale, point.length_2 * scale


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
    HYBRID = auto()  # Try both directions


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
                 optical_axis: np.ndarray,
                 threshold: float,
                 layer_config: dict = None):
        self.order = order
        self.trajectory = trajectory
        self.direction = direction
        self.placement = placement
        self.camera_pos = camera_pos
        self.optical_axis = optical_axis
        self.threshold = threshold
        self.layer_config = layer_config or {'count': 5, 'spacing': 0.2}

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
            # Global Centroid vector is simply the optical axis
            global_vec = self.optical_axis

        # 3. Process each point
        for pid in ordered_points:
            original_pos = graph.initial_positions[pid]
            point_data = graph.points[pid]

            # Determine Trajectory Vector (normalized)
            # This vector points AWAY from the camera by default
            if self.trajectory == TrajectoryType.POINT_SPECIFIC:
                vec = original_pos - self.camera_pos
                norm = np.linalg.norm(vec)
                traj_vec = vec / norm if norm > 0 else np.array([0, 0, 1])
            else:
                traj_vec = global_vec

            # Determine signs to check based on MoveDirection
            # +1.0 = Away, -1.0 = Towards
            search_directions = []

            if self.direction == MoveDirection.TOWARDS_CAMERA:
                search_directions = [-1.0]
            elif self.direction == MoveDirection.AWAY_FROM_CAMERA:
                search_directions = [1.0]
            elif self.direction == MoveDirection.HYBRID:
                # Prefer Away (positive) first? Or check both at each step.
                # The logic inside _find_position will handle stepping.
                # passing both 1.0 and -1.0 allows the searcher to check both.
                search_directions = [1.0, -1.0]

            # Find Position
            new_pos = self._find_position(pid, point_data, traj_vec, search_directions, final_positions, graph)
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
        L_new = L_original * (dist_new / dist_original) using optical axis distance.
        """
        scale = calculate_perspective_scale(current_pos, new_pos, self.camera_pos, self.optical_axis)

        # Check both lengths
        l1_new = point_data.length_1 * scale
        l2_new = point_data.length_2 * scale

        if l1_new > point_data.max_length_limit or l2_new > point_data.max_length_limit:
            # We don't log this as debug constantly because HYBRID/TOWARDS might trigger it often
            return False
        return True

    def _find_position(self, pid, point_data, traj_vec, search_directions, placed_positions, graph):
        original_pos = graph.initial_positions[pid]

        # Define search steps
        step_size = 0.01
        max_steps = 400  # Safety break

        logger.info(f"    Resolving Point {pid}. Trajectory: {traj_vec}, Directions: {search_directions}")

        if self.placement == PlacementType.MIN_DISTANCE:
            # Iterative search along vector
            for i in range(max_steps):
                current_dist = i * step_size

                # Check all allowed directions at this distance
                # For HYBRID, this means checking +dist and -dist at the same 'cost' (displacement)
                for sign in search_directions:
                    current_shift = current_dist * sign
                    candidate_pos = original_pos + (traj_vec * current_shift)

                    # 1. Check Perspective Limit
                    if not self._check_perspective_constraint(point_data, original_pos, candidate_pos):
                        # If moving away violates perspective, stop checking this direction entirely
                        # (further away will always violate)
                        if sign > 0:
                            pass  # We could remove 1.0 from search_directions to optimize, but logic is simple enough
                        continue

                    # 2. Check Interference (XY projection check)
                    if self._is_valid(candidate_pos, placed_positions, graph, current_id=pid):
                        if i > 0:
                            logger.debug(
                                f"      Found valid position for Point {pid} at step {i} (Shift: {current_shift:.2f})")
                        return candidate_pos

            logger.info(f"      Failed to find valid position for Point {pid}. Returning original.")
            return original_pos

        elif self.placement == PlacementType.LAYERS:
            # Construct layers
            for d in range(self.layer_config['count']):
                dist_abs = d * self.layer_config['spacing']

                # Check allowed directions for this layer
                for sign in search_directions:
                    shift = dist_abs * sign
                    candidate_pos = original_pos + (traj_vec * shift)

                    logger.debug(f"      Testing Layer {d} (Shift: {shift:.2f})")

                    if not self._check_perspective_constraint(point_data, original_pos, candidate_pos):
                        continue

                    if self._is_valid(candidate_pos, placed_positions, graph, current_id=pid):
                        logger.info(f"      Found valid position for Point {pid} at Layer {d} (Shift: {shift:.2f})")
                        return candidate_pos

            logger.info(f"      Failed to find valid position in layers for Point {pid}.")
            return original_pos

    def _is_valid(self, candidate_pos, placed_positions, graph, current_id=None):
        """Checks if candidate_pos interferes with any placed points in XY plane."""
        for neighbor_id, neighbor_pos in placed_positions.items():
            if check_overlap(candidate_pos, neighbor_pos, self.threshold) or \
                    check_downwash(candidate_pos, neighbor_pos, self.threshold):
                return False
        return True


# --- Main Solver Class ---

class ModularInterferenceSolver:
    def __init__(self,
                 graph: InterferenceGraph,
                 camera_pos: Tuple[float, float, float]):
        self.graph = graph
        self.camera_pos = np.array(camera_pos)
        # Compute optical axis once at initialization
        self.optical_axis = get_optical_axis(list(graph.points.values()), self.camera_pos)
        logger.info(f"Optical Axis Computed: {self.optical_axis}")

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
            optical_axis=self.optical_axis,
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


# --- Input / Output Utilities ---

def load_points_from_yaml(filepath: str) -> List[Point3D]:
    """Reads points from a YAML file."""
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} not found.")
        return []

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    points = []
    # Assuming YAML structure is: {'points': [ {id: 1, x: 1.0, ...}, ... ]}
    for p_data in data.get('points', []):
        points.append(Point3D(
            id=p_data['id'],
            x=p_data['x'],
            y=p_data['y'],
            z=p_data['z'],
            length_1=p_data.get('length_1', 1.0),
            length_2=p_data.get('length_2', 1.0),
            angle_1=p_data.get('angle_1', 0.0),
            angle_2=p_data.get('angle_2', 0.0),
            yaw=p_data.get('yaw', 0.0),  # Default Yaw is 0.0
            max_length_limit=p_data.get('max_length_limit', 10.0)
        ))
    logger.info(f"Loaded {len(points)} points from {filepath}")
    return points


def save_points_to_yaml(filepath: str, points: List[Point3D], positions: Dict[int, np.ndarray], camera_pos: np.ndarray,
                        optical_axis: np.ndarray):
    """Saves the updated points (position and scaled lengths) to a YAML file."""
    output_data = []
    for p in points:
        # Get new position (or original if not moved)
        new_pos = positions.get(p.id, np.array([p.x, p.y, p.z]))
        orig_pos = np.array([p.x, p.y, p.z])

        # Determine scale factor based on optical axis distance
        scale_factor = calculate_perspective_scale(orig_pos, new_pos, camera_pos, optical_axis)

        # Calculate new lengths based on perspective
        l1_new = p.length_1 * scale_factor
        l2_new = p.length_2 * scale_factor

        p_dict = {
            'id': p.id,
            'x': float(new_pos[0]),
            'y': float(new_pos[1]),
            'z': float(new_pos[2]),
            'length_1': float(l1_new),
            'length_2': float(l2_new),
            'angle_1': p.angle_1,
            'angle_2': p.angle_2,
            'yaw': p.yaw,  # Preserve yaw
            'max_length_limit': p.max_length_limit,
            'scale_factor': float(scale_factor)
        }
        output_data.append(p_dict)

    try:
        with open(filepath, 'w') as f:
            yaml.dump({'points': output_data}, f, sort_keys=False)
        logger.info(f"Saved {len(output_data)} updated points to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save output YAML: {e}")


def visualize_interference_graph(graph: InterferenceGraph,
                                 moved_indices: List[int],
                                 final_positions: Dict[int, np.ndarray],
                                 abstract_layout: bool = False):
    """
    Visualizes the initial interference graph.
    If abstract_layout is True, uses a 2D spring/force-directed layout.
    Otherwise, uses the actual 3D physical coordinates.
    Shows different edge colors for downwash vs. overlap conflicts.
    Marks nodes based on selection and movement status.
    """
    try:
        if abstract_layout:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
    except Exception as e:
        logger.info(f"Visualization setup failed: {e}")
        return

    moved_set = set(moved_indices)
    actual_moved = set()
    selected_not_moved = set()

    for pid in moved_set:
        orig = graph.initial_positions[pid]
        new = final_positions.get(pid, orig)
        if np.linalg.norm(new - orig) > 1e-6:
            actual_moved.add(pid)
        else:
            selected_not_moved.add(pid)

    if abstract_layout:
        # Generate abstract 2D layout
        edges_list = []
        for u in graph.nodes:
            for v in graph.get_neighbors(u):
                if u < v:
                    edges_list.append((u, v))

        if HAS_NX:
            G = nx.Graph()
            G.add_nodes_from(graph.nodes)
            G.add_edges_from(edges_list)
            pos = nx.shell_layout(G)
            # pos = nx.spring_layout(G)
            # pos = nx.kamada_kawai_layout(G)
        else:
            # Fallback to circle layout if networkx is missing
            pos = {}
            n = len(graph.nodes)
            for i, pid in enumerate(sorted(graph.nodes)):
                angle = 2 * math.pi * i / max(1, n)
                pos[pid] = np.array([math.cos(angle), math.sin(angle)])

        # Draw nodes in 2D
        for pid in graph.nodes:
            p = pos[pid]
            if pid in actual_moved:
                ax.scatter(p[0], p[1], c='green', marker='^', s=100, label='Selected & Moved', zorder=5)
            elif pid in selected_not_moved:
                ax.scatter(p[0], p[1], c='red', marker='X', s=100, label='Selected & NOT Moved', zorder=5)
            else:
                ax.scatter(p[0], p[1], c='blue', marker='o', s=50, label='Fixed (Not Selected)', zorder=5)
            ax.text(p[0], p[1] + 0.05, f" {pid}", fontsize=9, zorder=10)

        # Draw edges in 2D
        for u, v in edges_list:
            edge = tuple(sorted((u, v)))
            pos_u = pos[u]
            pos_v = pos[v]
            etype = graph.edge_types.get(edge, 'downwash')

            if etype == 'overlap':
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]],
                        c='red', linewidth=2, label='Overlap Edge', zorder=1)
            else:
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]],
                        c='orange', linewidth=2, linestyle='--', label='Downwash Edge', zorder=1)

        ax.set_title(
            f"Abstract Conflict Graph (2D Layout)\nConflicts: {graph.total_downwash} Downwash, {graph.total_overlap} Overlaps")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    else:
        # --- Original 3D drawing logic ---
        # Draw nodes
        for pid in graph.nodes:
            pos = graph.initial_positions[pid]
            if pid in actual_moved:
                ax.scatter(pos[0], pos[1], pos[2], c='green', marker='^', s=100, label='Selected & Moved')
            elif pid in selected_not_moved:
                ax.scatter(pos[0], pos[1], pos[2], c='red', marker='X', s=100, label='Selected & NOT Moved')
            else:
                ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='o', s=50, label='Fixed (Not Selected)')
            ax.text(pos[0], pos[1], pos[2] + 0.01, f" {pid}", fontsize=9)

        # Draw edges
        drawn_edges = set()
        for u in graph.nodes:
            for v in graph.get_neighbors(u):
                edge = tuple(sorted((u, v)))
                if edge not in drawn_edges:
                    drawn_edges.add(edge)
                    pos_u = graph.initial_positions[u]
                    pos_v = graph.initial_positions[v]
                    etype = graph.edge_types.get(edge, 'downwash')

                    if etype == 'overlap':
                        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                                c='red', linewidth=2, label='Overlap Edge')
                    else:
                        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                                c='orange', linewidth=2, linestyle='--', label='Downwash Edge')

        ax.set_title(
            f"Initial Conflict Graph (3D Actual)\nConflicts: {graph.total_downwash} Downwash, {graph.total_overlap} Overlaps")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_aspect('equal')
        ax.view_init(elev=0, azim=0)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if args.save_viz:
        plt.savefig(args.viz_graph_output_file, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_conflict_bar_chart(graph: InterferenceGraph):
    """
    Visualizes the number of downwash and overlap conflicts per point using a grouped bar chart.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
    except Exception as e:
        logger.info(f"Visualization setup failed: {e}")
        return

    nodes = sorted(graph.nodes)
    df_data = []
    for pid in nodes:
        df_data.append({'LightBender ID': str(pid), 'Conflicts': graph.node_downwashes[pid], 'Type': 'Downwash'})
        df_data.append({'LightBender ID': str(pid), 'Conflicts': graph.node_overlaps[pid], 'Type': 'Overlap'})
    
    df = pd.DataFrame(df_data)

    sns.barplot(data=df, x='LightBender ID', y='Conflicts', hue='Type', ax=ax, palette=['#ffb347', '#ff6961'])
    sns.despine(left=True, bottom=True)
    ax.tick_params(axis='x', labelsize=6)
    
    ax.set_ylabel('Number of Conflicts', rotation=0, ha='left', labelpad=15)
    ax.yaxis.set_label_coords(-0.03, 1.02)

    plt.tight_layout()
    ax.legend(title=None)

    if args.save_viz:
        plt.savefig(args.viz_bar_output_file, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_solution_2d(graph: InterferenceGraph,
                          moved_indices: List[int],
                          final_positions: Dict[int, np.ndarray],
                          camera_pos: np.ndarray):
    """
    Visualizes the before and after states of the points in the XY plane.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [1, 3]})
    except Exception as e:
        logger.info(f"Visualization setup failed: {e}")
        return

    # Extract data
    all_ids = graph.nodes
    orig_coords = np.array([graph.initial_positions[pid][:2] for pid in all_ids])
    final_coords = np.array([final_positions[pid][:2] for pid in all_ids])
    moved_set = set(moved_indices)

    # Distinguish actually-moved from selected-but-not-moved (same logic as graph visualizer)
    actual_moved = set()
    selected_not_moved = set()
    for pid in moved_set:
        orig = graph.initial_positions[pid]
        new = final_positions.get(pid, orig)
        if np.linalg.norm(new - orig) > 1e-6:
            actual_moved.add(pid)
        else:
            selected_not_moved.add(pid)

    # --- Plot 1: Before ---
    ax1.set_title(f"Before Resolution ({len(moved_indices)} selected)")

    for i, pid in enumerate(all_ids):
        color = 'red' if pid in moved_set else 'blue'
        ax1.scatter(orig_coords[i, 0], orig_coords[i, 1], c=color, s=40, zorder=3)
        circle = plt.Circle(orig_coords[i], graph.threshold / 2, color=color, fill=False, alpha=0.2)
        ax1.add_patch(circle)
        # ax1.text(orig_coords[i, 0] + 0.1, orig_coords[i, 1] + 0.1, str(pid), fontsize=8)

    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    sns.despine(ax=ax1)

    # --- Plot 2: After ---
    ax2.set_title("After Resolution")
    ax2.plot(camera_pos[0], camera_pos[1], 'k*', markersize=15)
    ax2.text(camera_pos[0], camera_pos[1] - 0.15, 'Viewpoint', ha='center', va='top')

    for i, pid in enumerate(all_ids):
        if pid in actual_moved:
            color, marker = 'orange', '^'
        elif pid in selected_not_moved:
            color, marker = 'red', 'X'
        else:
            color, marker = 'blue', 'o'
        ax2.scatter(final_coords[i, 0], final_coords[i, 1], c=color, marker=marker, s=60, zorder=3)
        circle = plt.Circle(final_coords[i], graph.threshold / 2, color=color, fill=False, alpha=0.2)
        ax2.add_patch(circle)
        if pid in actual_moved:
            start = graph.initial_positions[pid][:2]
            end = final_positions[pid][:2]
            ax2.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                      head_width=0.02, color='black', alpha=0.5, length_includes_head=True, zorder=200)

    ax2.scatter([], [], c='orange', marker='^', s=60, label='Selected & Moved')
    ax2.scatter([], [], c='red', marker='X', s=60, label='Selected & NOT Moved')
    ax2.scatter([], [], c='blue', marker='o', s=60, label='Fixed (Not Selected)')

    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    sns.despine(ax=ax2)

    plt.tight_layout()

    if args.save_viz:
        plt.savefig(args.viz_2d_output_file, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_3d_structure(points: List[Point3D], final_positions: Dict[int, np.ndarray], camera_pos: np.ndarray,
                           optical_axis: np.ndarray):
    """
    Visualizes points and their attached lines in 3D for both before and after states.
    Shows the actual length (scaled by perspective) vs the max length limit.
    """
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    except Exception as e:
        logger.error(f"3D Visualization failed: {e}")
        return

    # Plot Camera
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='k', marker='*', s=200, label='Camera')

    # Plot Optical Axis (optional, for debugging)
    # ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], optical_axis[0], optical_axis[1], optical_axis[2], length=2.0, color='r', alpha=0.5, label="Optical Axis")

    # Helper to calculate vector tip based on body frame rules + Yaw
    def get_line_tip(cx, cy, cz, length, angle_deg, yaw_deg):
        # 1. Local Body Frame (Rotation axis -X, Zero Angle +Y)
        rad = np.radians(angle_deg)
        dy_local = length * np.cos(rad)
        dz_local = -length * np.sin(rad)
        dx_local = 0.0

        # 2. Apply Yaw Rotation (around +Z)
        rad_yaw = np.radians(yaw_deg)
        cos_y = np.cos(rad_yaw)
        sin_y = np.sin(rad_yaw)

        # RotZ * [dx, dy, dz]^T
        dx_rot = dx_local * cos_y - dy_local * sin_y
        dy_rot = dx_local * sin_y + dy_local * cos_y
        dz_rot = dz_local

        return np.array([cx + dx_rot, cy + dy_rot, cz + dz_rot])

    def draw_structure(origin, length_1, length_2, max_len, angle_1, angle_2, yaw, color_main, alpha_main, style_main):
        """Draws the lines for a point, showing filled length vs max length."""

        # Line 1
        # Max limit (Container)
        tip1_max = get_line_tip(origin[0], origin[1], origin[2], max_len, angle_1, yaw)
        ax.plot([origin[0], tip1_max[0]], [origin[1], tip1_max[1]], [origin[2], tip1_max[2]],
                color=color_main, alpha=0.15, linewidth=1, linestyle=':')

        # Actual Length (Filler)
        tip1_act = get_line_tip(origin[0], origin[1], origin[2], length_1, angle_1, yaw)
        ax.plot([origin[0], tip1_act[0]], [origin[1], tip1_act[1]], [origin[2], tip1_act[2]],
                color=color_main, alpha=alpha_main, linewidth=2, linestyle=style_main)

        # Line 2
        # Max limit (Container)
        tip2_max = get_line_tip(origin[0], origin[1], origin[2], max_len, angle_2, yaw)
        ax.plot([origin[0], tip2_max[0]], [origin[1], tip2_max[1]], [origin[2], tip2_max[2]],
                color=color_main, alpha=0.15, linewidth=1, linestyle=':')

        # Actual Length (Filler)
        tip2_act = get_line_tip(origin[0], origin[1], origin[2], length_2, angle_2, yaw)
        ax.plot([origin[0], tip2_act[0]], [origin[1], tip2_act[1]], [origin[2], tip2_act[2]],
                color=color_main, alpha=alpha_main, linewidth=2, linestyle=style_main)

    for p in points:
        orig_pos = np.array([p.x, p.y, p.z])
        new_pos = final_positions.get(p.id, orig_pos)

        has_moved = np.linalg.norm(orig_pos - new_pos) > 1e-6

        # --- Plot Original State (Ghosted) ---
        if has_moved:
            ax.scatter(p.x, p.y, p.z, c='gray', marker='o', alpha=0.3)

            # Draw Original Structure
            draw_structure(orig_pos, p.length_1, p.length_2, p.max_length_limit,
                           p.angle_1, p.angle_2, p.yaw, 'gray', 0.3, '--')

            # Arrow from Old to New
            ax.plot([p.x, new_pos[0]], [p.y, new_pos[1]], [p.z, new_pos[2]],
                    color='black', alpha=0.5, linestyle=':', linewidth=1)

        # --- Plot Final State (Solid) ---
        color = 'orange' if has_moved else 'blue'
        ax.scatter(new_pos[0], new_pos[1], new_pos[2], c=color, marker='o', s=40,
                   label=f"Point {p.id}" if p.id == 0 else "")
        ax.text(new_pos[0], new_pos[1], new_pos[2], f" {p.id}", fontsize=9)

        # Calculate new physical lengths based on perspective scaling using optical axis
        l1_final, l2_final = calculate_scaled_lengths(p, new_pos, camera_pos, optical_axis)

        # Line 1 (Green-ish) - Max Limit then Actual
        tip1_max = get_line_tip(new_pos[0], new_pos[1], new_pos[2], p.max_length_limit, p.angle_1, p.yaw)
        ax.plot([new_pos[0], tip1_max[0]], [new_pos[1], tip1_max[1]], [new_pos[2], tip1_max[2]],
                color='green', alpha=0.2, linewidth=1, linestyle=':')

        tip1_act = get_line_tip(new_pos[0], new_pos[1], new_pos[2], l1_final, p.angle_1, p.yaw)
        ax.plot([new_pos[0], tip1_act[0]], [new_pos[1], tip1_act[1]], [new_pos[2], tip1_act[2]],
                color='green', alpha=0.9, linewidth=2, linestyle='-')

        # Line 2 (Magenta-ish) - Max Limit then Actual
        tip2_max = get_line_tip(new_pos[0], new_pos[1], new_pos[2], p.max_length_limit, p.angle_2, p.yaw)
        ax.plot([new_pos[0], tip2_max[0]], [new_pos[1], tip2_max[1]], [new_pos[2], tip2_max[2]],
                color='magenta', alpha=0.2, linewidth=1, linestyle=':')

        tip2_act = get_line_tip(new_pos[0], new_pos[1], new_pos[2], l2_final, p.angle_2, p.yaw)
        ax.plot([new_pos[0], tip2_act[0]], [new_pos[1], tip2_act[1]], [new_pos[2], tip2_act[2]],
                color='magenta', alpha=0.9, linewidth=2, linestyle='-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D LightBender Structure (Dotted=Max Limit, Solid=Scaled Length)")
    ax.set_aspect('equal')
    # ax.view_init(elev=0, azim=0)

    if args.save_viz:
        plt.savefig(args.viz_3d_output_file, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular Interference Solver")

    # Files
    parser.add_argument("--input_file", type=str, default="initial_layout.yaml", help="Path to input YAML file")
    parser.add_argument("--output_file", type=str, default="feasible_layout.yaml", help="Path to output YAML file")
    parser.add_argument("--viz_2d_output_file", type=str, default="2d_viz.png",
                        help="Path to output 2d visualization file")
    parser.add_argument("--viz_3d_output_file", type=str, default="3d_viz.png",
                        help="Path to output 3d visualization file")
    parser.add_argument("--viz_graph_output_file", type=str, default="graph_viz.png",
                        help="Path to output graph visualization file")
    parser.add_argument("--viz_bar_output_file", type=str, default="bar_viz.png",
                        help="Path to output conflict bar chart file")

    # Parameters
    parser.add_argument("--threshold", type=float, default=0.16, help="Interference threshold distance")
    parser.add_argument("--camera_pos", type=float, nargs=3, default=[2.3, 0.0, 0.8], help="Camera position (x y z)")

    # Algorithm Config Enums
    parser.add_argument("--selection_method", type=str, default="GREEDY_MAX_DEGREE",
                        choices=[e.name for e in SelectionMethod], help="Selection method")
    parser.add_argument("--resolution_order", type=str, default="MAX_DEGREE",
                        choices=[e.name for e in ResolutionOrder], help="Resolution order")
    parser.add_argument("--trajectory_type", type=str, default="POINT_SPECIFIC",
                        choices=[e.name for e in TrajectoryType], help="Trajectory type")
    parser.add_argument("--move_direction", type=str, default="HYBRID",
                        choices=[e.name for e in MoveDirection], help="Movement direction")
    parser.add_argument("--placement_type", type=str, default="MIN_DISTANCE",
                        choices=[e.name for e in PlacementType], help="Placement type")

    parser.add_argument("--no_viz", action='store_true', help="Disable visualization")
    parser.add_argument("--save_viz", action='store_true', help="Save visualization as files")
    parser.add_argument("--abstract_graph", action='store_true', help="Use abstract 2D layout for graph visualization")
    parser.add_argument("--csv", action='store_true', help="Output metrics as CSV to stdout")

    args = parser.parse_args()

    # Convert args to Enums
    sel_method = SelectionMethod[args.selection_method]
    res_order = ResolutionOrder[args.resolution_order]
    traj_type = TrajectoryType[args.trajectory_type]
    move_dir = MoveDirection[args.move_direction]
    place_type = PlacementType[args.placement_type]

    CAMERA_POS = tuple(args.camera_pos)

    # 1. Generate Sample Data if input file does not exist
    if not os.path.exists(args.input_file):
        logger.info(f"Input file {args.input_file} not found. Generating sample data at this path.")
        sample_points = []
        np.random.seed(101)
        for i in range(10):  # Generated 10 points default
            sample_points.append({
                'id': i,
                'x': float(np.random.rand() * 10),
                'y': float(np.random.rand() * 10),
                'z': float(np.random.rand() * 5 + 10),
                'length_1': 1.5,
                'length_2': 1.0,
                'angle_1': float(np.random.rand() * 90),
                'angle_2': float(np.random.rand() * 90 + 180),
                'yaw': float(np.random.rand() * 360),  # Random Yaw
                'max_length_limit': 4.0
            })
        try:
            with open(args.input_file, 'w') as f:
                yaml.dump({'points': sample_points}, f)
        except IOError as e:
            logger.error(f"Failed to create sample input file: {e}")
            exit(1)

    # 2. Load Data
    points_data = load_points_from_yaml(args.input_file)

    if not points_data:
        logger.error("No points loaded. Exiting.")
        exit(1)

    # 3. Build Graph
    graph = InterferenceGraph(points_data, args.threshold)

    # 4. Initialize Solver
    solver = ModularInterferenceSolver(graph, CAMERA_POS)

    # 5. Run with Configuration
    logger.info(f"--- Configuration: {sel_method.name} + {traj_type.name} + {move_dir.name} + {place_type.name} ---")
    moved, positions = solver.solve(
        selection_method=sel_method,
        resolution_order=res_order,
        trajectory_type=traj_type,
        move_direction=move_dir,
        placement_type=place_type
    )

    # 6. Validation Output & Metrics
    logger.info(f"Graph stats: {len(graph.nodes)} nodes, threshold {graph.threshold}")

    moved_distances = []

    # Retrieve computed optical axis
    OPTICAL_AXIS = solver.optical_axis

    for pid in moved:
        orig = graph.initial_positions[pid]
        new = positions[pid]
        dist_3d = np.linalg.norm(new - orig)

        # Track distance if actually moved
        if dist_3d > 1e-6:
            moved_distances.append(dist_3d)

        # Calculate new lengths for reporting
        p = graph.points[pid]
        l1_new, l2_new = calculate_scaled_lengths(p, new, np.array(CAMERA_POS), OPTICAL_AXIS)

        logger.info(f"Point {pid} moved {dist_3d:.2f} units to {new}")
        logger.info(
            f"  Perspective Scale: L1 {p.length_1:.2f}->{l1_new:.2f}, L2 {p.length_2:.2f}->{l2_new:.2f} (Max: {p.max_length_limit})")

    # Metrics Calculation
    num_selected = len(moved)
    num_moved = len(moved_distances)
    num_downwashes = graph.total_downwash
    num_overlaps = graph.total_overlap

    # Per-node conflict stats (initial)
    init_node_total = {pid: graph.node_overlaps[pid] + graph.node_downwashes[pid] for pid in graph.points}
    init_min_total = min(init_node_total.values()) if init_node_total else 0
    init_max_total = max(init_node_total.values()) if init_node_total else 0
    init_min_dw = min(graph.node_downwashes.values()) if graph.node_downwashes else 0
    init_max_dw = max(graph.node_downwashes.values()) if graph.node_downwashes else 0
    init_min_col = min(graph.node_overlaps.values()) if graph.node_overlaps else 0
    init_max_col = max(graph.node_overlaps.values()) if graph.node_overlaps else 0

    import copy
    final_points_data = copy.deepcopy(points_data)
    for p_dict in final_points_data:
        pid = p_dict.id
        if pid in positions:
            p_dict.x = float(positions[pid][0])
            p_dict.y = float(positions[pid][1])
            p_dict.z = float(positions[pid][2])

    final_graph = InterferenceGraph(final_points_data, args.threshold)
    unresolved_downwashes = final_graph.total_downwash
    unresolved_overlaps = final_graph.total_overlap

    # Per-node conflict stats (final)
    final_node_total = {pid: final_graph.node_overlaps[pid] + final_graph.node_downwashes[pid] for pid in final_graph.points}
    final_min_total = min(final_node_total.values()) if final_node_total else 0
    final_max_total = max(final_node_total.values()) if final_node_total else 0
    final_min_dw = min(final_graph.node_downwashes.values()) if final_graph.node_downwashes else 0
    final_max_dw = max(final_graph.node_downwashes.values()) if final_graph.node_downwashes else 0
    final_min_col = min(final_graph.node_overlaps.values()) if final_graph.node_overlaps else 0
    final_max_col = max(final_graph.node_overlaps.values()) if final_graph.node_overlaps else 0

    if num_moved > 0:
        avg_dist = float(np.mean(moved_distances))
        min_dist = float(np.min(moved_distances))
        max_dist = float(np.max(moved_distances))
    else:
        avg_dist = 0.0
        min_dist = 0.0
        max_dist = 0.0

    if args.csv:
        # CSV format: Downwashes,Overlaps,UnresolvedDownwashes,UnresolvedOverlaps,PointsSelected,PointsMoved,AvgDist,MinDist,MaxDist
        print(
            f"{num_downwashes},{num_overlaps},{unresolved_downwashes},{unresolved_overlaps},{init_min_dw},{init_max_dw},{init_min_col},{init_max_col},{init_min_total},{init_max_total},{final_min_dw},{final_max_dw},{final_min_col},{final_max_col},{final_min_total},{final_max_total},{num_selected},{num_moved},{avg_dist:.4f},{min_dist:.4f},{max_dist:.4f}")
    else:
        print("=" * 40)
        print("METRICS SUMMARY")
        print(f"Initial Downwash conflicts:    {num_downwashes}  (per-node min/max: {init_min_dw}/{init_max_dw})")
        print(f"Initial Overlap Conflicts:   {num_overlaps}  (per-node min/max: {init_min_col}/{init_max_col})")
        print(f"Initial Total:                 (per-node min/max: {init_min_total}/{init_max_total})")
        print(f"Unresolved Downwashes:         {unresolved_downwashes}  (per-node min/max: {final_min_dw}/{final_max_dw})")
        print(f"Unresolved Overlaps:         {unresolved_overlaps}  (per-node min/max: {final_min_col}/{final_max_col})")
        print(f"Unresolved Total:              (per-node min/max: {final_min_total}/{final_max_total})")
        print(f"LightBenders Selected to Move: {num_selected}")
        print(f"LightBenders Actually Moved:   {num_moved}")
        print(f"Travel Distance - Avg:         {avg_dist:.4f}")
        print(f"Travel Distance - Min:         {min_dist:.4f}")
        print(f"Travel Distance - Max:         {max_dist:.4f}")
        print("=" * 40)

    # 7. Save Results
    save_points_to_yaml(args.output_file, points_data, positions, np.array(CAMERA_POS), OPTICAL_AXIS)

    # 8. Visualize Results (Optional)
    if not args.no_viz:
        # Interference Graph Visualization
        visualize_interference_graph(graph, moved, positions, abstract_layout=args.abstract_graph)
        # Conflict Bar Chart
        visualize_conflict_bar_chart(graph)
        # 2D Before/After
        visualize_solution_2d(graph, moved, positions, np.array(CAMERA_POS))
        # 3D Structure Visualization
        visualize_3d_structure(points_data, positions, np.array(CAMERA_POS), OPTICAL_AXIS)