#!/usr/bin/env python3
"""Find camera-facing two-rod layouts with identical perspective silhouettes.

YAML/JSON points use the same x/y/z, yaw (degrees), angle_1/2 (degrees),
length_1/2 (metres) convention as deconflict.py. No Blender dependency.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


def rotation(yaw, pitch=0, roll=0):
    y, p, r = np.radians([yaw, pitch, roll])
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def endpoints(point):
    R = rotation(point['yaw'], point.get('pitch', 0), point.get('roll', 0))
    hinge = np.array([point[k] for k in ('x', 'y', 'z')]) + R @ np.array(point.get('light_module_offset', [0, 0, 0]))
    tips = []
    for i in (1, 2):
        a = math.radians(point[f'angle_{i}'])
        tips.append(hinge + point[f'length_{i}'] * (R @ np.array([0, math.cos(a), -math.sin(a)])))
    return hinge, np.array(tips), R[:, 0]


def _solve_point(point, camera, margin_degrees=10, min_depth_scale=0.1, reverse_plane=False):
    """Return an independent result; failed/unchanged results retain input pose.

    Exactness concerns continuous rod silhouettes, not individual LED samples,
    brightness, mutual occlusion, or animation between independently solved poses.
    """
    result = copy.deepcopy(point)
    kind = point.get('type', 'H').removeprefix('TYPE_')
    if kind not in ('H', 'V'):
        return dict(result, status='unsupported', reason='Only straight two-rod H/V modules are supported')
    C = np.asarray(camera, dtype=float)
    H, tips, normal = endpoints(point)
    v = C - H
    distance = np.linalg.norm(v)
    if distance < 1e-9:
        return dict(result, status='infeasible', reason='Camera coincides with module hinge')
    cosine = float(normal @ v / distance)
    result['facing_cosine_before'] = cosine
    result['facing_away'] = cosine < 0
    threshold = math.sin(math.radians(margin_degrees))
    if cosine >= threshold:
        return dict(result, status='unchanged', facing_cosine_after=cosine, source_rods=[1, 2], depth_scale=1.0)
    if np.linalg.norm(v[:2]) < 1e-9:
        return dict(result, status='infeasible', reason='Yaw-only lighting plane cannot face a camera directly above/below')
    beta = math.atan2(v[1], v[0])
    delta = math.radians(point['yaw']) - beta
    folded = math.atan2(math.sin(delta), abs(math.cos(delta)))
    # Account for elevation: a horizontal normal can achieve at most this cosine.
    horizontal_fraction = np.linalg.norm(v[:2]) / distance
    if horizontal_fraction < threshold:
        return dict(result, status='infeasible', reason='Requested facing margin requires module tilt')
    limit = math.acos(min(1, threshold / horizontal_fraction))
    new_yaw = (math.radians(point['yaw']) + math.pi if reverse_plane
               else beta + max(-limit, min(limit, folded)))
    # Unwrap near the source yaw, without altering the physical orientation.
    new_yaw += math.tau * round((math.radians(point['yaw']) - new_yaw) / math.tau)
    R = rotation(math.degrees(new_yaw))
    n, u = R[:, 0], R[:, 1]
    numer = float(n @ (H - C))
    projected = []
    for tip in tips:
        ray = tip - C
        denom = float(n @ ray)
        if abs(denom) < 1e-10 or numer / denom <= 0:
            return dict(result, status='infeasible', reason='Endpoint ray is parallel to plane or intersects behind camera')
        projected.append(C + (numer / denom) * ray - H)
    lengths = [float(np.linalg.norm(d)) for d in projected]
    ranges = [(0, 180), (180, 360)] if kind == 'H' else [(90, 270), (270, 450)]
    maxima = point.get('max_lengths', [point.get('max_length_limit', 0.156)] * 2)
    choices = []
    for order in ((0, 1), (1, 0)):
        angles = []
        for slot, source in enumerate(order):
            d = projected[source]
            if lengths[source] < 1e-10:
                a = float(point[f'angle_{slot+1}'])
            else:
                a = math.degrees(math.atan2(-d[2], float(d @ u)))
            candidates = [a + k*360 for k in range(-2, 3) if ranges[slot][0]-1e-7 <= a+k*360 <= ranges[slot][1]+1e-7]
            if not candidates:
                break
            angles.append(min(candidates, key=lambda x: abs(x-point[f'angle_{slot+1}'])))
        if len(angles) != 2:
            continue
        scale = min([1.0] + [maxima[j]/lengths[i] for j, i in enumerate(order) if lengths[i] > 1e-10])
        if scale < min_depth_scale:
            continue
        choices.append((scale, order, angles))
    if not choices:
        return dict(result, status='infeasible', reason='Servo ranges or minimum depth scale prevent an exact solution')
    scale, order, angles = max(choices, key=lambda c: (c[0], c[1] == (0, 1)))
    new_hinge = C + scale * (H - C)
    origin = new_hinge - R @ np.array(point.get('light_module_offset', [0, 0, 0]))
    result.update(zip(('x', 'y', 'z'), map(float, origin)))
    result.update(yaw=math.degrees(new_yaw), roll=0.0, pitch=0.0,
                  angle_1=angles[0], angle_2=angles[1],
                  length_1=scale*lengths[order[0]], length_2=scale*lengths[order[1]],
                  source_rods=[i+1 for i in order], depth_scale=scale,
                  facing_cosine_after=float(n @ (C-new_hinge)/np.linalg.norm(C-new_hinge)),
                  strategy='reverse_normal' if reverse_plane else 'ray_plane_refit', status='corrected')
    return result


def solve_point(point, camera, margin_degrees=10, min_depth_scale=0.1):
    """Prefer reversing a level module's normal in its existing lighting plane.

    This can preserve the physical light segments as well as their images.
    Near edge-on poses and tilted inputs use the general ray/plane refit.
    """
    if point.get('type', 'H').removeprefix('TYPE_') in ('H', 'V'):
        H, _, n = endpoints(point)
        v = np.asarray(camera, dtype=float) - H
        distance = np.linalg.norm(v)
        if (distance > 1e-9 and abs(point.get('roll', 0)) < 1e-9
                and abs(point.get('pitch', 0)) < 1e-9
                and n @ v / distance <= -math.sin(math.radians(margin_degrees))):
            reversed_result = _solve_point(point, camera, margin_degrees, min_depth_scale, True)
            if reversed_result['status'] == 'corrected':
                return reversed_result
    return _solve_point(point, camera, margin_degrees, min_depth_scale)


def solve_layout(document):
    """Solve independently: includes explicit status for every point."""
    camera = document['camera']
    margin = document.get('margin_degrees', 10)
    minimum = document.get('min_depth_scale', 0.1)
    if len(camera) != 3 or not all(math.isfinite(float(x)) for x in camera):
        raise ValueError('camera must contain three finite numbers')
    if not 0 < margin < 90 or not 0 < minimum <= 1:
        raise ValueError('margin_degrees must be in (0,90); min_depth_scale in (0,1]')
    points = document['points']
    ids = [p['id'] for p in points]
    if len(set(ids)) != len(ids):
        raise ValueError('point IDs must be unique')
    for p in points:
        values = [p[k] for k in ('x', 'y', 'z', 'yaw', 'angle_1', 'angle_2', 'length_1', 'length_2')]
        values += [p.get('roll', 0), p.get('pitch', 0)]
        offset = p.get('light_module_offset', [0, 0, 0])
        maxima = p.get('max_lengths', [p.get('max_length_limit', 0.156)]*2)
        if len(offset) != 3 or len(maxima) != 2 or not all(math.isfinite(float(x)) for x in values+list(offset)+list(maxima)):
            raise ValueError(f"Invalid geometry for point {p['id']}")
        if min(p['length_1'], p['length_2']) < 0 or min(maxima) <= 0:
            raise ValueError('rod lengths must be nonnegative and limits positive')
    return dict(document, points=[solve_point(p, camera, margin, minimum) for p in points])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_file', required=True, type=Path)
    parser.add_argument('--output_file', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.input_file.suffix.lower() == '.json':
            document = json.loads(args.input_file.read_text())
        else:
            import yaml
            document = yaml.safe_load(args.input_file.read_text())
        result = solve_layout(document)
        # JSON is also valid YAML; Blender can parse this without PyYAML.
        args.output_file.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'camera_facing: {error}\n')


if __name__ == '__main__':
    main()
