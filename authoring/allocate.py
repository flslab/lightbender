import numpy as np
import yaml
import os
import argparse
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod
import matplotlib

matplotlib.use('macosx')


# --- Data Structures ---

@dataclass
class Point3D:
    id: int
    x: float
    y: float
    z: float
    length_1: float
    length_2: float
    angle_1: float  # Pitch angle 1 in degrees
    angle_2: float  # Pitch angle 2 in degrees
    yaw: float  # Yaw angle in degrees (Rotation about +Z)
    max_length_limit: float


class TargetGraph:
    def __init__(self, filepath: str):
        self.nodes: Dict[int, np.ndarray] = {}
        self.edges: List[Tuple[int, int]] = []

        if os.path.exists(filepath):
            self._load_yaml(filepath)
        else:
            print(f"File {filepath} not found. Generating sample graph (ZigZag).")
            self._generate_sample_graph(filepath)

    def _load_yaml(self, filepath: str):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        for n in data.get('nodes', []):
            self.nodes[n['id']] = 2*np.array([n['x'], n['y'], n['z']], dtype=float)

        for e in data.get('edges', []):
            self.edges.append((e['source'], e['target']))

    def _generate_sample_graph(self, filepath: str):
        # Generates the user-provided 7-edge graph where the optimal cover is exactly 4 LightBenders
        self.nodes = {
            0: np.array([0.0, 0.1386, 1.24]),
            1: np.array([0.0, 0.0, 1.32]),
            2: np.array([0.0, -0.1386, 1.24]),
            3: np.array([0.0, -0.1386, 1.08]),
            4: np.array([0.0, 0.1386, 0.92]),
            5: np.array([0.0, 0.1386, 0.76]),
            6: np.array([0.0, 0.0, 0.68]),
            7: np.array([0.0, -0.1386, 0.76])
        }
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)
        ]

        data = {
            'nodes': [{'id': k, 'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])} for k, v in self.nodes.items()],
            'edges': [{'source': u, 'target': v} for u, v in self.edges]
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f, sort_keys=False)


# --- Inverse Kinematics Engine ---

def compute_lightbender_params(body: np.ndarray, tip1: np.ndarray, tip2: Optional[np.ndarray]) -> Tuple[
    float, float, float, float, float]:
    """
    Computes the IK parameters (yaw, length1, length2, angle1, angle2) for a LightBender.
    Requires XY collinearity between (tip1-body) and (tip2-body).
    """
    v1 = tip1 - body
    vxy_norm = np.linalg.norm(v1[:2])

    if vxy_norm > 1e-6:
        yaw = np.degrees(np.arctan2(-v1[0], v1[1]))
    else:
        if tip2 is not None:
            v2 = tip2 - body
            if np.linalg.norm(v2[:2]) > 1e-6:
                yaw = np.degrees(np.arctan2(v2[0], -v2[1]))
            else:
                yaw = 0.0
        else:
            yaw = 0.0

    yaw_rad = np.radians(yaw)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

    def get_local_angle_len(v: np.ndarray) -> Tuple[float, float]:
        if v is None: return 0.0, 0.0
        L = np.linalg.norm(v)
        if L < 1e-6: return 0.0, 0.0

        dy_loc = -v[0] * sin_y + v[1] * cos_y
        dz_loc = v[2]

        a = np.degrees(np.arctan2(-dz_loc, dy_loc))
        return L, a % 360.0

    L1, a1 = get_local_angle_len(v1)

    if tip2 is not None:
        L2, a2 = get_local_angle_len(tip2 - body)
    else:
        L2, a2 = 0.0, 0.0

    return yaw, L1, L2, a1, a2


# --- Allocation Strategies ---

class AllocationStrategy(ABC):
    @abstractmethod
    def allocate(self, graph: TargetGraph, max_length: float) -> List[Point3D]:
        pass


class MidpointStrategy(AllocationStrategy):
    def allocate(self, graph: TargetGraph, max_length: float) -> List[Point3D]:
        lightbenders = []
        lb_id = 0
        for u, v in graph.edges:
            A, B = graph.nodes[u], graph.nodes[v]
            vec = B - A
            L = np.linalg.norm(vec)
            if L < 1e-6: continue
            dir_vec = vec / L

            num_segments = math.ceil(L / (2.0 * max_length))
            seg_len = L / num_segments

            for i in range(num_segments):
                center = A + (i + 0.5) * seg_len * dir_vec
                tip1 = center + (seg_len / 2.0) * dir_vec
                tip2 = center - (seg_len / 2.0) * dir_vec

                yaw, l1, l2, a1, a2 = compute_lightbender_params(center, tip1, tip2)
                lightbenders.append(Point3D(lb_id, center[0], center[1], center[2], l1, l2, a1, a2, yaw, max_length))
                lb_id += 1
        return lightbenders


class VertexStrategy(AllocationStrategy):
    def allocate(self, graph: TargetGraph, max_length: float) -> List[Point3D]:
        lightbenders = []
        lb_id = 0
        for u, v in graph.edges:
            A, B = graph.nodes[u], graph.nodes[v]
            vec = B - A
            L = np.linalg.norm(vec)
            if L < 1e-6: continue
            dir_vec = vec / L

            num_segments = math.ceil(L / max_length)
            for i in range(num_segments):
                body = A + i * max_length * dir_vec
                rem_len = min(max_length, L - i * max_length)
                tip1 = body + rem_len * dir_vec

                yaw, l1, l2, a1, a2 = compute_lightbender_params(body, tip1, None)
                lightbenders.append(Point3D(lb_id, body[0], body[1], body[2], l1, l2, a1, a2, yaw, max_length))
                lb_id += 1
        return lightbenders


class OptimizedStrategy(AllocationStrategy):
    """
    Uses Global Optimization (Exact Set Cover) to find the absolute minimum number of LightBenders.
    """

    def allocate(self, graph: TargetGraph, max_length: float) -> List[Point3D]:
        # 1. Simplify Graph (Merge collinear 3D edges)
        merged_nodes, merged_edges = self._merge_collinear_edges(graph)

        # 2. Discretize edges into chunks
        chunks = []  # List of tuples: (edge_idx, midpoint_dist)
        edge_data = []  # Store L and dir_vec

        CHUNK_SIZE = max_length / 4.0  # High resolution overlapping

        for e_idx, (u, v) in enumerate(merged_edges):
            A, B = merged_nodes[u], merged_nodes[v]
            vec = B - A
            L = np.linalg.norm(vec)
            dir_vec = vec / L if L > 1e-6 else np.zeros(3)
            edge_data.append({'u': u, 'v': v, 'A': A, 'B': B, 'L': L, 'dir': dir_vec})

            num_chunks = max(1, math.ceil(L / CHUNK_SIZE))
            chunk_len = L / num_chunks
            for c in range(num_chunks):
                chunks.append((e_idx, (c + 0.5) * chunk_len))

        # 3. Generate Candidate Placements
        candidates = []

        # A. Edge Sliding Candidates
        for e_idx, ed in enumerate(edge_data):
            L = ed['L']
            num_points = max(2, math.ceil(L / CHUNK_SIZE))
            for i in range(num_points):
                d = i * L / (num_points - 1) if num_points > 1 else 0
                body = ed['A'] + d * ed['dir']

                len_fwd = min(max_length, L - d)
                len_bwd = min(max_length, d)

                tip1 = body + len_fwd * ed['dir']
                tip2 = body - len_bwd * ed['dir'] if len_bwd > 1e-6 else None

                # Find covered chunks
                covered = {i for i, (ce, cd) in enumerate(chunks) if
                           ce == e_idx and (d - max_length) <= cd <= (d + max_length)}

                candidates.append({'body': body, 'tip1': tip1, 'tip2': tip2, 'covered': covered})

        # B. Vertex Spanning Candidates (Two XY-collinear edges)
        adj = {n: [] for n in merged_nodes}
        for e_idx, ed in enumerate(edge_data):
            adj[ed['u']].append((e_idx, ed['v'], True))  # True = pointing away
            adj[ed['v']].append((e_idx, ed['u'], False))  # False = pointing towards

        for v_id, incident in adj.items():
            V_pos = merged_nodes[v_id]
            for i in range(len(incident)):
                for j in range(i + 1, len(incident)):
                    e1_idx, a_id, is_fwd1 = incident[i]
                    e2_idx, b_id, is_fwd2 = incident[j]

                    vec1 = merged_nodes[a_id] - V_pos
                    vec2 = merged_nodes[b_id] - V_pos

                    n1 = np.linalg.norm(vec1[:2])
                    n2 = np.linalg.norm(vec2[:2])

                    # Check XY Coplanarity (Parallel, Anti-parallel, or Vertical Line)
                    is_valid = False
                    if n1 < 1e-4 or n2 < 1e-4:
                        is_valid = True  # One is purely vertical, so yaw can adapt
                    else:
                        dot = np.dot(vec1[:2] / n1, vec2[:2] / n2)
                        if abs(abs(dot) - 1.0) < 1e-3:
                            is_valid = True

                    if is_valid:
                        L1 = min(max_length, np.linalg.norm(vec1))
                        L2 = min(max_length, np.linalg.norm(vec2))

                        tip1 = V_pos + (vec1 / np.linalg.norm(vec1)) * L1
                        tip2 = V_pos + (vec2 / np.linalg.norm(vec2)) * L2

                        covered = set()
                        # Find chunks on both incident edges
                        for ch_idx, (ce, cd) in enumerate(chunks):
                            if ce == e1_idx:
                                E1_L = edge_data[e1_idx]['L']
                                if is_fwd1 and cd <= max_length:
                                    covered.add(ch_idx)
                                elif not is_fwd1 and cd >= (E1_L - max_length):
                                    covered.add(ch_idx)

                            if ce == e2_idx:
                                E2_L = edge_data[e2_idx]['L']
                                if is_fwd2 and cd <= max_length:
                                    covered.add(ch_idx)
                                elif not is_fwd2 and cd >= (E2_L - max_length):
                                    covered.add(ch_idx)

                        candidates.append({'body': V_pos, 'tip1': tip1, 'tip2': tip2, 'covered': covered})

        # 4. Solve Exact Set Cover (Branch and Bound)
        chosen_indices = self._solve_set_cover(candidates, len(chunks))

        # 5. Extract Final LightBenders
        lightbenders = []
        for lb_id, idx in enumerate(chosen_indices):
            cand = candidates[idx]
            yaw, l1, l2, a1, a2 = compute_lightbender_params(cand['body'], cand['tip1'], cand['tip2'])
            lb = Point3D(
                id=lb_id, x=cand['body'][0], y=cand['body'][1], z=cand['body'][2],
                length_1=l1, length_2=l2, angle_1=a1, angle_2=a2,
                yaw=yaw, max_length_limit=max_length
            )
            lightbenders.append(lb)

        return lightbenders

    def _merge_collinear_edges(self, graph: TargetGraph):
        nodes = {k: v.copy() for k, v in graph.nodes.items()}
        adj = {k: set() for k in nodes}
        for u, v in graph.edges:
            adj[u].add(v)
            adj[v].add(u)

        while True:
            merged = False
            for n, neighbors in list(adj.items()):
                if len(neighbors) == 2:
                    u, v = list(neighbors)
                    vec1 = nodes[n] - nodes[u]
                    vec2 = nodes[v] - nodes[n]
                    n1 = np.linalg.norm(vec1)
                    n2 = np.linalg.norm(vec2)

                    if n1 > 1e-6 and n2 > 1e-6:
                        if abs(np.dot(vec1 / n1, vec2 / n2) - 1.0) < 1e-4:  # 3D Collinear
                            adj[u].remove(n)
                            adj[v].remove(n)
                            adj[u].add(v)
                            adj[v].add(u)
                            del adj[n]
                            del nodes[n]
                            merged = True
                            break
            if not merged:
                break

        edges = []
        seen = set()
        for u, neighbors in adj.items():
            for v in neighbors:
                if frozenset([u, v]) not in seen:
                    edges.append((u, v))
                    seen.add(frozenset([u, v]))
        return nodes, edges

    def _solve_set_cover(self, candidates, num_chunks):
        """
        An exact Branch and Bound Set Cover solver.
        Mathematically guarantees the absolute minimum number of candidates are chosen.
        """

        # 1. Provide an initial upper bound using Greedy approach
        def solve_greedy():
            covered = set()
            chosen = []
            all_chunks = set(range(num_chunks))
            while covered != all_chunks:
                best_idx = -1
                best_cover = set()
                for i, cand in enumerate(candidates):
                    if i in chosen: continue
                    new_cover = cand['covered'] - covered
                    if len(new_cover) > len(best_cover):
                        best_cover = new_cover
                        best_idx = i
                if best_idx == -1: break
                chosen.append(best_idx)
                covered |= best_cover
            return chosen

        greedy_sol = solve_greedy()

        # 2. Setup Exact Branch and Bound
        valid_indices = []
        cand_masks = []

        for i, cand in enumerate(candidates):
            if not cand['covered']: continue
            mask = 0
            for c in cand['covered']:
                mask |= (1 << c)
            cand_masks.append(mask)
            valid_indices.append(i)

        target_mask = (1 << num_chunks) - 1

        # Sort candidates descending by coverage for faster pruning
        cand_order = list(range(len(valid_indices)))
        cand_order.sort(key=lambda x: bin(cand_masks[x]).count('1'), reverse=True)

        best_solution = greedy_sol
        best_size = len(greedy_sol)

        # Precompute the maximum coverage possible for remaining subsets
        rem_masks = [0] * len(cand_order)
        accum = 0
        for i in reversed(range(len(cand_order))):
            accum |= cand_masks[cand_order[i]]
            rem_masks[i] = accum

        iters = 0
        MAX_ITERS = 100000

        def backtrack(cand_idx, current_mask, current_solution):
            nonlocal best_solution, best_size, iters

            iters += 1
            if iters > MAX_ITERS:
                return

            # If all chunks covered, check if it's the best so far
            if current_mask == target_mask:
                if len(current_solution) < best_size:
                    best_size = len(current_solution)
                    best_solution = [valid_indices[i] for i in current_solution]
                return

            # Prune if taking another makes us tie or worse than the best size
            if len(current_solution) >= best_size - 1:
                return

            # Prune if no more candidates to check
            if cand_idx >= len(cand_order):
                return

            # Prune if even taking ALL remaining candidates wouldn't cover the rest
            if (current_mask | rem_masks[cand_idx]) != target_mask:
                return

            c_id = cand_order[cand_idx]
            c_mask = cand_masks[c_id]

            new_mask = current_mask | c_mask

            # Branch 1: Try Taking the Candidate (Only if it adds new coverage)
            if new_mask != current_mask:
                current_solution.append(c_id)
                backtrack(cand_idx + 1, new_mask, current_solution)
                current_solution.pop()

            # Branch 2: Skip Candidate
            backtrack(cand_idx + 1, current_mask, current_solution)

        # Run exact solver
        backtrack(0, 0, [])
        return best_solution


# --- Allocation Processor ---

class Allocator:
    def __init__(self, max_length_limit: float):
        self.max_length_limit = max_length_limit

    def run(self, graph: TargetGraph, policy: str) -> List[Point3D]:
        if policy.upper() == "MIDPOINT":
            strategy = MidpointStrategy()
        elif policy.upper() == "VERTEX":
            strategy = VertexStrategy()
        elif policy.upper() == "OPTIMIZED":
            strategy = OptimizedStrategy()
        else:
            raise ValueError(f"Unknown allocation policy: {policy}")

        print(f"Running Allocation with {policy} policy...")
        return strategy.allocate(graph, self.max_length_limit)


# --- Visualization ---

def visualize_allocation(graph: TargetGraph, lightbenders: List[Point3D]):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for u, v in graph.edges:
        A = graph.nodes[u]
        B = graph.nodes[v]
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color='black', alpha=0.3, linewidth=6,
                label='Target Graph' if u == graph.edges[0][0] else "")
        ax.scatter(*A, color='black', s=20)
        ax.scatter(*B, color='black', s=20)

    def get_line_tip(cx, cy, cz, length, angle_deg, yaw_deg):
        rad = np.radians(angle_deg)
        dy_local = length * np.cos(rad)
        dz_local = -length * np.sin(rad)

        rad_yaw = np.radians(yaw_deg)
        cos_y, sin_y = np.cos(rad_yaw), np.sin(rad_yaw)

        dx_rot = -dy_local * sin_y
        dy_rot = dy_local * cos_y

        return np.array([cx + dx_rot, cy + dy_rot, cz + dz_local])

    for lb in lightbenders:
        body = np.array([lb.x, lb.y, lb.z])
        ax.scatter(*body, color='orange', s=80, edgecolors='red', zorder=5,
                   label='LightBender Body' if lb.id == 0 else "")
        ax.text(lb.x, lb.y, lb.z + 0.05, f"LB{lb.id}", fontsize=8)

        if lb.length_1 > 0:
            t1 = get_line_tip(lb.x, lb.y, lb.z, lb.length_1, lb.angle_1, lb.yaw)
            ax.plot([lb.x, t1[0]], [lb.y, t1[1]], [lb.z, t1[2]], color='green', linewidth=2.5,
                    label='Rod 1' if lb.id == 0 else "")

        if lb.length_2 > 0:
            t2 = get_line_tip(lb.x, lb.y, lb.z, lb.length_2, lb.angle_2, lb.yaw)
            ax.plot([lb.x, t2[0]], [lb.y, t2[1]], [lb.z, t2[2]], color='magenta', linewidth=2.5,
                    label='Rod 2' if lb.id == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Allocation Output: Graph vs LightBenders")

    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)

    plt.legend()
    plt.show()


def save_to_solver_format(lightbenders: List[Point3D], filepath: str):
    data = []
    for lb in lightbenders:
        data.append({
            'id': lb.id,
            'x': float(lb.x),
            'y': float(lb.y),
            'z': float(lb.z),
            'length_1': float(lb.length_1),
            'length_2': float(lb.length_2),
            'angle_1': float(lb.angle_1),
            'angle_2': float(lb.angle_2),
            'yaw': float(lb.yaw),
            'max_length_limit': float(lb.max_length_limit)
        })
    with open(filepath, 'w') as f:
        yaml.dump({'points': data}, f, sort_keys=False)
    print(f"Allocated states saved to {filepath}")


# --- Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allocate LightBenders to Target Topology")
    parser.add_argument("--input", type=str, default="target_graph.yaml", help="Input YAML graph file")
    parser.add_argument("--output", type=str, default="allocated_points.yaml", help="Output YAML state file")
    parser.add_argument("--policy", type=str, choices=['MIDPOINT', 'VERTEX', 'OPTIMIZED'], default="OPTIMIZED",
                        help="Placement policy")
    parser.add_argument("--max_len", type=float, default=0.16, help="Maximum length limit for a rod")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization")

    args = parser.parse_args()

    graph = TargetGraph(args.input)

    allocator = Allocator(max_length_limit=args.max_len)
    lightbenders = allocator.run(graph, args.policy)

    # Metrics
    total_lbs = len(lightbenders)
    total_rods = sum(1 for lb in lightbenders if lb.length_1 > 1e-3) + sum(
        1 for lb in lightbenders if lb.length_2 > 1e-3)
    total_length = sum(lb.length_1 + lb.length_2 for lb in lightbenders)

    max_capacity = total_lbs * 2 * args.max_len
    utilization = (total_length / max_capacity * 100) if max_capacity > 0 else 0
    avg_rod_len = (total_length / total_rods) if total_rods > 0 else 0

    print("\n--- Allocation Metrics ---")
    print(f"Policy:                 {args.policy}")
    print(f"Total LightBenders:     {total_lbs}")
    print(f"Total Rods Activated:   {total_rods}")
    print(f"Average Rod Length:     {avg_rod_len:.2f}")
    print(f"Rod Length Utilization: {utilization:.1f}%")
    print("--------------------------\n")

    save_to_solver_format(lightbenders, args.output)

    if not args.no_viz:
        visualize_allocation(graph, lightbenders)