"""Perspective occlusion against the full Blender drone mesh.

The cached CAD-derived triangle surface is mounted at the mechanical hinge.
Weighted midpoint samples on illuminated rods estimate hidden projected length;
triangle ray intersections preserve the frame's gaps and nonconvex geometry.
Module and body mesh offsets are world-frame vectors, as in Point3D.
"""
import math
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from mesh_occlusion import TriangleMesh

import numpy as np


@lru_cache(maxsize=1)
def load_body_mesh():
    """Load the exact triangulation exported from the add-on's full model."""
    candidates = [Path(__file__).parent/'assets', Path(__file__).parent.parent/'assets']
    asset = next((folder/'lightbender_occlusion.npz' for folder in candidates
                  if (folder/'lightbender_occlusion.npz').is_file()), None)
    if asset is None:
        raise ValueError('Drone occlusion mesh is missing; run authoring/fusion/export_occlusion_mesh.py with Blender')
    with np.load(asset, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
        source = asset.with_name('lightbender_full.blend')
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != metadata['source_sha256']:
            raise ValueError('Drone occlusion mesh is stale; re-export it from lightbender_full.blend with Blender')
        mesh = TriangleMesh(data['vertices'], data['faces'])
    mesh.asset_metadata = metadata
    return mesh


def rays_blocked_by_body(camera, targets, origin, yaw_degrees, mesh=None):
    """Intersect finite camera rays with the full mesh at its mechanical hinge."""
    mesh = load_body_mesh() if mesh is None else mesh
    angle = math.radians(yaw_degrees)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    camera_local = (np.asarray(camera)-origin) @ rotation
    target_local = (np.atleast_2d(targets)-origin) @ rotation
    return mesh.blocked(camera_local, target_local)


class BodyOcclusionEvaluator:
    """Reusable initial-layout evaluator; positions may be a partial mapping."""
    def __init__(self, points, initial_positions, camera_pos, optical_axis, samples_per_rod=33, mesh=None):
        self.mesh = load_body_mesh() if mesh is None else mesh
        self.points = points if isinstance(points, dict) else {p.id: p for p in points}
        self.initial = {pid: np.asarray(pos, dtype=float) for pid, pos in initial_positions.items()}
        self.camera = np.asarray(camera_pos, dtype=float)
        self.axis = np.array(optical_axis, dtype=float, copy=True)
        self.axis /= np.linalg.norm(self.axis)
        right = np.cross(self.axis, [0., 0., 1.])
        if np.linalg.norm(right) < 1e-10:
            right = np.cross(self.axis, [0., 1., 0.])
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.axis)
        if samples_per_rod < 1:
            raise ValueError('samples_per_rod must be positive')
        self.samples_per_rod = samples_per_rod
        self.rods = []
        for pid, point in self.points.items():
            intervals = point.lit_intervals
            if intervals is None:
                intervals = [[[0., point.length_1]], [[0., point.length_2]]]
            for rod, spans in enumerate(intervals):
                if point.rod_directions is not None:
                    direction = np.asarray(point.rod_directions[rod], dtype=float)
                else:
                    a = math.radians(getattr(point, f'angle_{rod+1}'))
                    yaw = math.radians(point.yaw)
                    direction = np.array([-math.sin(yaw)*math.cos(a),
                                          math.cos(yaw)*math.cos(a), -math.sin(a)])
                for j, (start, end) in enumerate(spans):
                    if end <= start:
                        continue
                    if point.lit_colors is not None:
                        color = point.lit_colors[rod][j]
                        if color is not None and list(color) == [0, 0, 0]:
                            continue
                    boundaries = np.linspace(start, end, samples_per_rod+1)
                    self.rods.append((pid, rod, direction, boundaries))

    def evaluate(self, positions=None, details=False):
        positions = {**self.initial, **(positions or {})}
        samples, weights, owners = [], [], []
        for pid, rod, direction, distances in self.rods:
            offset = np.asarray(self.points[pid].module_offset)
            hinge = np.asarray(positions[pid])+offset
            old_depth = (self.initial[pid]+offset-self.camera) @ self.axis
            depth = (hinge-self.camera) @ self.axis
            if old_depth <= 1e-9 or depth <= 1e-9:
                continue
            boundaries = hinge + (distances*(depth/old_depth))[:, None]*direction
            rays = boundaries-self.camera
            z = rays @ self.axis
            if np.any(z <= 1e-9):
                continue
            uv = np.stack((rays @ self.right, rays @ self.up), axis=1)/z[:, None]
            lengths = np.linalg.norm(np.diff(uv, axis=0), axis=1)
            samples.extend((boundaries[:-1]+boundaries[1:])*.5)
            weights.extend(lengths)
            owners.extend([pid]*len(lengths))
        if not samples:
            return {'occluded_projected_length': 0., 'total_projected_lit_length': 0.,
                    'occluded_fraction': 0., 'blockers': [], 'samples_per_rod': self.samples_per_rod} if details else 0.
        samples, weights = np.asarray(samples), np.asarray(weights)
        blocked = np.zeros(len(samples), dtype=bool)
        blockers = []
        for pid, point in self.points.items():
            offset = getattr(point, 'body_mesh_offset', None)
            offset = point.module_offset if offset is None else offset
            origin = np.asarray(positions[pid])+np.asarray(offset)
            hit = rays_blocked_by_body(self.camera, samples, origin, point.yaw, self.mesh)
            blocked |= hit
            if details and np.any(hit):
                victims = []
                for target in self.points:
                    selected = hit & np.array([owner == target for owner in owners])
                    amount = float(weights[selected].sum())
                    if amount > 1e-12:
                        victims.append({'drone_id': target, 'occluded_projected_length': amount})
                blockers.append({'drone_id': pid, 'occluded_projected_length': float(weights[hit].sum()),
                                 'victims': victims})
        amount = float(weights[blocked].sum())
        if not details:
            return amount
        total = float(weights.sum())
        return {'occluded_projected_length': amount, 'total_projected_lit_length': total,
                'occluded_fraction': amount/total if total else 0., 'blockers': blockers,
                'samples_per_rod': self.samples_per_rod}


def body_occlusion_report(points, initial_positions, positions, camera_pos, optical_axis,
                          samples_per_rod=33, mesh=None):
    """Return sampled initial-body visibility metrics for a proposed placement.

    ``initial_positions`` belongs to the tentative layout; ``positions`` is the
    staggered drone-origin mapping. Rod lengths follow the same perspective
    scaling as Stagger. The hidden length uses a unit-depth camera image plane.
    """
    evaluator = BodyOcclusionEvaluator(points, initial_positions, camera_pos, optical_axis,
                                        samples_per_rod, mesh=mesh)
    report = evaluator.evaluate(positions, details=True)
    geometry = dict(model='triangle_mesh', mesh_origin='mechanical module hinge',
                    vertices=evaluator.mesh.vertex_count, triangles=evaluator.mesh.triangle_count)
    geometry.update(getattr(evaluator.mesh, 'asset_metadata', {}))
    return dict(report, geometry=geometry,
                method='Finite camera-ray/triangle intersections weighted by projected lit-segment length')
