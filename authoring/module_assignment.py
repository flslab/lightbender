"""Assign straight two-rod layouts to H/V modules without changing lit geometry.

Horizontal rods prefer V servos (away from H's 0/180-degree end stops), while
vertical rods prefer H servos (away from V's 90/270-degree end stops). A preference
never overrides a physical angle/length constraint or available inventory.
"""
from __future__ import annotations

import math

NEAR_AXIS_DEGREES = 15.0
SERVO_RANGES = {'H': ((0., 180.), (180., 360.)),
                'V': ((90., 270.), (270., 450.))}


def preferred_module_type(point, near_axis_degrees=NEAR_AXIS_DEGREES):
    """Return V for nearly horizontal lines, H for nearly vertical, else None.

    Both lit arms must be near the same axis; opposing directions count as the
    same line. An unlit arm does not affect this classification.
    """
    if not math.isfinite(near_axis_degrees) or not 0 <= near_axis_degrees < 45:
        raise ValueError('near_axis_degrees must be finite and in [0,45)')
    angles = [float(point[f'angle_{i}']) % 180 for i in (1, 2)
              if float(point[f'length_{i}']) > 1e-10]
    if not angles:
        return None
    if all(min(a, 180-a) <= near_axis_degrees+1e-9 for a in angles):
        return 'V'
    if all(abs(a-90) <= near_axis_degrees+1e-9 for a in angles):
        return 'H'
    return None


def compatible_assignments(point, near_axis_degrees=NEAR_AXIS_DEGREES, rod_limits=None):
    """Return preferred-first (kind, source_order, legal_servo_angles) tuples.

    source_order[j] identifies which original rod is carried by physical slot j.
    Angles preserve each lit rod's direction modulo 360. Unlit arms are placed
    within their hardware range; swapping also requires swapping lengths/colors.
    rod_limits, when supplied, are physical length limits for slots 0 and 1.
    """
    lengths = [float(point[f'length_{i}']) for i in (1, 2)]
    angles = [float(point[f'angle_{i}']) for i in (1, 2)]
    if not all(math.isfinite(v) for v in lengths+angles) or min(lengths) < 0:
        raise ValueError('module angles/lengths must be finite and lengths nonnegative')
    if rod_limits is not None:
        if len(rod_limits) != 2 or not all(math.isfinite(v) and v > 0 for v in rod_limits):
            raise ValueError('rod_limits must contain two positive finite lengths')
    preferred = preferred_module_type(point, near_axis_degrees)
    result = []
    for kind in sorted(SERVO_RANGES, key=lambda k: (k != preferred, k)):
        for order in ((0, 1), (1, 0)):
            selected = []
            for slot, source in enumerate(order):
                if rod_limits is not None and lengths[source] > rod_limits[slot]+1e-10:
                    break
                lo, hi = SERVO_RANGES[kind][slot]
                a = angles[source] % 360
                candidates = [a+360*k for k in (-1, 0, 1)
                              if lo-1e-9 <= a+360*k <= hi+1e-9]
                if not candidates and lengths[source] <= 1e-10:
                    candidates = [(lo+hi)/2]
                if not candidates:
                    break
                selected.append(max(lo, min(hi, min(candidates, key=lambda v: abs(v-angles[source])))))
            if len(selected) == 2:
                result.append((kind, order, selected))
    return result


def assign_modules(points, inventory=None, near_axis_degrees=NEAR_AXIS_DEGREES,
                   rod_limits=None, allow_extra=False):
    """Choose one assignment per point, reserving inventory across the layout.

    inventory maps H/V to lists of drone IDs and is never mutated. Among feasible
    assignments, minimize missed axis preferences. With allow_extra=True, first
    minimize generated extra drones, then missed preferences. Result order matches
    input order; infeasible geometry or insufficient strict inventory raises.
    """
    options = [compatible_assignments(p, near_axis_degrees, rod_limits) for p in points]
    for index, choices in enumerate(options):
        if not choices:
            raise ValueError(f'placement point {points[index].get("id", index)}: no compatible H/V module')
    if inventory is None:
        return [choices[0] for choices in options]
    capacities = {kind: len(inventory.get(kind, [])) for kind in SERVO_RANGES}
    n = len(points)
    # With only two types, H count determines V count at each layer. This exact
    # small DP prevents an ambiguous early point taking a later point's only type.
    states = {0: (0, [])}
    for index, choices in enumerate(options):
        first_by_type = {}
        for choice in choices:
            first_by_type.setdefault(choice[0], choice)
        preferred = preferred_module_type(points[index], near_axis_degrees)
        next_states = {}
        for h_count, (cost, selected) in states.items():
            for kind, choice in first_by_type.items():
                next_h = h_count + (kind == 'H')
                used = next_h if kind == 'H' else index+1-next_h
                extra = used > capacities[kind]
                if extra and not allow_extra:
                    continue
                new_cost = cost + int(preferred is not None and kind != preferred) + (n+1)*extra
                current = next_states.get(next_h)
                if current is None or new_cost < current[0]:
                    next_states[next_h] = (new_cost, selected+[choice])
        states = next_states
        if not states:
            raise ValueError('insufficient compatible H/V module inventory for the layout')
    return min(states.values(), key=lambda state: state[0])[1]
