"""Finite camera-ray occlusion against an actual triangle mesh using NumPy.

A cached median-split BVH is only an acceleration structure: leaves test the
original triangles, retaining openings and non-convex geometry. Queries use
mesh-local coordinates. The caller applies aircraft pose and mounting transforms.
"""
from __future__ import annotations

import numpy as np


class TriangleMesh:
    """Reusable two-sided triangle intersector with a cached bounding hierarchy."""

    def __init__(self, vertices, faces, leaf_size=16):
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3
                or not np.all(np.isfinite(vertices))):
            raise ValueError('vertices must be a finite N x 3 array')
        if (faces.ndim != 2 or faces.shape[1] != 3
                or not np.issubdtype(faces.dtype, np.integer)):
            raise ValueError('faces must be an integer M x 3 array')
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('face indices are outside the vertex array')
        if isinstance(leaf_size, bool) or not isinstance(leaf_size, (int, np.integer)) or leaf_size < 1:
            raise ValueError('leaf_size must be a positive integer')
        triangles = vertices[faces]
        self.vertex_count, self.triangle_count = len(vertices), len(faces)
        self.v0 = triangles[:, 0].copy()
        self.e1 = triangles[:, 1]-triangles[:, 0]
        self.e2 = triangles[:, 2]-triangles[:, 0]
        self.edge_scale = np.linalg.norm(self.e1, axis=1)*np.linalg.norm(self.e2, axis=1)
        self._triangle_min = triangles.min(axis=1)
        self._triangle_max = triangles.max(axis=1)
        centers = (self._triangle_min+self._triangle_max)*.5
        order = np.arange(len(faces))
        nodes = []

        def build(start, end):
            indices = order[start:end]
            lower = self._triangle_min[indices].min(axis=0)
            upper = self._triangle_max[indices].max(axis=0)
            index = len(nodes)
            nodes.append(None)
            if end-start <= leaf_size:
                nodes[index] = (lower, upper, -1, -1, start, end)
            else:
                axis = int(np.argmax(np.ptp(centers[indices], axis=0)))
                middle = (end-start)//2
                partition = np.argpartition(centers[indices, axis], middle)
                order[start:end] = indices[partition]
                split = start+middle
                left, right = build(start, split), build(split, end)
                nodes[index] = (lower, upper, left, right, start, end)
            return index

        if len(faces):
            build(0, len(faces))
        self._nodes = nodes
        self._order = order
        # Per-triangle bounds are no longer needed after building the BVH.
        del self._triangle_min, self._triangle_max

    @staticmethod
    def _box_hits(camera, directions, lower, upper, endpoint_tolerance):
        """Slab intersection of finite rays; zero direction components are valid."""
        parallel = abs(directions) < 1e-15
        inv = np.divide(1., directions, out=np.zeros_like(directions), where=~parallel)
        a, b = (lower-camera)*inv, (upper-camera)*inv
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        lo[parallel], hi[parallel] = -np.inf, np.inf
        possible = ~np.any(parallel & ((camera < lower) | (camera > upper)), axis=1)
        near = np.maximum(np.max(lo, axis=1), endpoint_tolerance)
        far = np.minimum(np.min(hi, axis=1), 1.-endpoint_tolerance)
        return possible & (near <= far)

    def _leaf_hits(self, camera, directions, triangle_indices, endpoint_tolerance):
        """Möller–Trumbore ray tests, vectorized over rays and leaf triangles."""
        e1, e2 = self.e1[triangle_indices], self.e2[triangle_indices]
        offset = camera-self.v0[triangle_indices]
        h = np.cross(directions[:, None, :], e2[None, :, :])
        determinant = np.einsum('tj,rtj->rt', e1, h)
        threshold = 1e-12 * np.linalg.norm(directions, axis=1)[:, None] * self.edge_scale[triangle_indices]
        valid = abs(determinant) > threshold
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
        u = np.einsum('tj,rtj->rt', offset, h)*inverse
        q = np.cross(offset, e1)
        v = (directions @ q.T)*inverse
        t = np.einsum('tj,tj->t', e2, q)[None, :]*inverse
        barycentric_tolerance = 1e-10
        return np.any(valid & (u >= -barycentric_tolerance) & (v >= -barycentric_tolerance)
                      & (u+v <= 1.+barycentric_tolerance)
                      & (t > endpoint_tolerance) & (t < 1.-endpoint_tolerance), axis=1)

    def blocked(self, camera, targets, endpoint_tolerance=1e-9):
        """Which camera-to-target segments hit a triangle before their endpoint?

        Returns one Boolean per target, including a single XYZ target. Surface
        contact at the camera or at the target is excluded; triangle edge hits
        count. Coplanar rays are not surface crossings. Geometry is two-sided.
        ``endpoint_tolerance`` is a fraction of each finite segment's length.
        """
        camera = np.asarray(camera, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if targets.shape == (3,):
            targets = targets[None, :]
        if (camera.shape != (3,) or targets.ndim != 2 or targets.shape[1] != 3
                or not np.all(np.isfinite(camera)) or not np.all(np.isfinite(targets))):
            raise ValueError('camera must be XYZ and targets a finite N x 3 array')
        if not np.isfinite(endpoint_tolerance) or not 0 <= endpoint_tolerance < .5:
            raise ValueError('endpoint_tolerance must be finite and in [0,.5)')
        result = np.zeros(len(targets), dtype=bool)
        if not self._nodes or not len(targets):
            return result
        directions = targets-camera
        stack = [(0, np.arange(len(targets)))]
        while stack:
            node_id, rays = stack.pop()
            rays = rays[~result[rays]]
            if not len(rays):
                continue
            lower, upper, left, right, start, end = self._nodes[node_id]
            rays = rays[self._box_hits(camera, directions[rays], lower, upper, endpoint_tolerance)]
            if not len(rays):
                continue
            if left < 0:
                hits = self._leaf_hits(camera, directions[rays], self._order[start:end], endpoint_tolerance)
                result[rays[hits]] = True
            else:
                stack.append((right, rays))
                stack.append((left, rays))
        return result
