"""Analytic dark-rod occlusion checks for the designated perspective observer.

Screen-space bounding boxes reject unrelated rods before segment/capsule and
perspective-depth tests. Repairs transfer source colors onto foreground spans and extinguish the covered source.
"""
from dataclasses import dataclass, replace
import numpy as np


def merge_intervals(intervals, length):
    result = []
    for a, b in sorted(intervals):
        a, b = max(0., float(a)), min(float(length), float(b))
        if b <= a + 1e-10:
            continue
        if result and a <= result[-1][1] + 1e-10:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def _band(lo, hi, value, slope, lower, upper):
    if abs(slope) < 1e-14:
        return (lo, hi) if lower <= value <= upper else None
    a, b = sorted(((lower-value)/slope, (upper-value)/slope))
    lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if hi >= lo else None


def _capsule_intervals(a, b, c, d, radius):
    """Parameters on a--b inside the projected capsule around c--d."""
    v, w = b-a, d-c
    vv, ww = v@v, w@w
    if ww > 1e-24:
        length = np.sqrt(ww)
        normal = np.array([-w[1], w[0]]) / length
        interval = _band(0., 1., (a-c)@w/ww, v@w/ww, 0., 1.)
        if interval:
            interval = _band(*interval, (a-c)@normal, v@normal, -radius, radius)
            if interval:
                yield interval
    for endpoint in (c, d):
        offset = a-endpoint
        if vv < 1e-24:
            if offset@offset <= radius*radius:
                yield 0., 1.
            continue
        beta, gamma = offset@v, offset@offset-radius*radius
        discriminant = beta*beta-vv*gamma
        if discriminant >= 0:
            root = np.sqrt(discriminant)
            lo, hi = max(0., (-beta-root)/vv), min(1., (-beta+root)/vv)
            if hi >= lo:
                yield lo, hi


@dataclass
class Segment:
    owner: int
    rod: int
    uv: np.ndarray
    depth: np.ndarray
    radius: float
    start: float = 0.
    end: float = 0.
    color: object = None

    @property
    def bounds(self):
        return np.r_[self.uv.min(axis=0)-self.radius, self.uv.max(axis=0)+self.radius]


def collinear_range(dark, lit):
    """Projected overlap on the dark segment; None means non-collinear."""
    c,d = dark.uv
    a,b = lit.uv
    w = d-c
    ww = w@w
    if ww < 1e-20:
        return None
    normal = np.array([-w[1],w[0]])/np.sqrt(ww)
    if max(abs((a-c)@normal), abs((b-c)@normal)) > 1e-8:
        return None
    lo,hi = sorted(((a-c)@w/ww, (b-c)@w/ww))
    return max(0.,lo), min(1.,hi)


def blocks(dark, lit):
    """Does a dark physical rod pass in front of any of the lit centerline?"""
    a, b = lit.uv
    c, d = dark.uv
    w = d-c
    ww = w@w
    overlap = collinear_range(dark, lit)
    if overlap is not None and overlap[1] <= overlap[0]+1e-10:
        return False  # Collinear segments merely sharing an endpoint.
    for lo, hi in _capsule_intervals(a, b, c, d, dark.radius):
        # Reciprocal depth is affine in projected segment coordinates. The
        # midpoint also covers degenerate projected segments/end caps.
        for t in (lo, (lo+hi)*.5, hi):
            s = np.clip(((a+(b-a)*t-c)@w)/ww, 0., 1.) if ww > 1e-24 else 0.
            front = 1./((1-s)/dark.depth[0]+s/dark.depth[1])
            back = 1./((1-t)/lit.depth[0]+t/lit.depth[1])
            # Coplanar shared endpoints are not depth occlusions.
            if front < back-0.003:
                return True
    return False


def front_ranges(dark, lit, lo, hi, in_front=True):
    """Clip a repair to the part actually in front, including tilted rods."""
    a,b = lit.uv
    c,d = dark.uv
    v = b-a
    vv = v@v
    q0 = (c-a)@v/vv if vv > 1e-24 else 0.
    dq = (d-c)@v/vv if vv > 1e-24 else 0.
    edges = [lo,hi]
    if abs(dq) > 1e-14:
        edges += [s for q in (0.,1.) if lo < (s:=(q-q0)/dq) < hi]
    for start,end in zip(sorted(edges),sorted(edges)[1:]):
        def separation(s):
            q = np.clip(q0+dq*s,0.,1.)
            difference = (1-s)/dark.depth[0]+s/dark.depth[1] - (1-q)/lit.depth[0]-q/lit.depth[1]
            return difference if in_front else -difference
        y0,y1 = separation(start),separation(end)
        if max(y0,y1) <= 1e-12:
            continue
        if min(y0,y1) < 0:
            crossing = start-y0*(end-start)/(y1-y0)
            if y0 < 0:
                start = crossing
            else:
                end = crossing
        yield start,end


def physical_distance(segment, t):
    alpha = (t/segment.depth[1])/((1-t)/segment.depth[0]+t/segment.depth[1])
    return float(segment.start+alpha*(segment.end-segment.start))


def slice_segment(segment, start, end, color=None):
    alpha = (np.array([start,end])-segment.start)/(segment.end-segment.start)
    depth = (1-alpha)*segment.depth[0]+alpha*segment.depth[1]
    uv = ((1-alpha[:,None])*segment.uv[0]*segment.depth[0]
          +alpha[:,None]*segment.uv[1]*segment.depth[1])/depth[:,None]
    return replace(segment,uv=uv,depth=depth,start=start,end=end,
                   radius=segment.radius*min(segment.depth)/min(depth),color=color)


def subtract_intervals(start, end, removed):
    cursor = start
    for a,b in merge_intervals(removed,end):
        if a > cursor+1e-10:
            yield cursor,a
        cursor = max(cursor,b)
    if end > cursor+1e-10:
        yield cursor,end


def covered_intervals(target, sources, bounds):
    """Physical target spans hidden by a nearer source; all geometry is analytic."""
    box = target.bounds
    mask = ((bounds[:,0] <= box[2]) & (bounds[:,2] >= box[0])
            & (bounds[:,1] <= box[3]) & (bounds[:,3] >= box[1]))
    for index in np.flatnonzero(mask):
        front = sources[index]
        if front.owner == target.owner or min(front.depth) >= max(target.depth):
            continue
        overlap = collinear_range(target,front)
        ranges = [overlap] if overlap is not None else _capsule_intervals(
            *target.uv,*front.uv,front.radius)
        for lo,hi in ranges:
            if hi <= lo+1e-10:
                continue
            for lo,hi in front_ranges(target,front,lo,hi,in_front=False):
                a,b = physical_distance(target,lo),physical_distance(target,hi)
                if b > a+1e-10:
                    yield a,b


class RodVisibilityIndex:
    def __init__(self, points, original_positions, camera, axis, positions=None):
        self.points = points
        self.original_positions = original_positions
        self.camera = np.asarray(camera, dtype=float)
        self.axis = np.asarray(axis, dtype=float)
        up = np.array([0., 0., 1.])
        if abs(up@self.axis) > .99:
            up = np.array([0., 1., 0.])
        right = np.cross(self.axis, up)
        right /= np.linalg.norm(right)
        self.basis = np.array([right, np.cross(right, self.axis)])
        self.entries = {}
        self.dirty = True
        for pid, pos in (positions or original_positions).items():
            self.update(pid, pos)

    def geometry(self, pid, position):
        p = self.points[pid]
        offset = np.asarray(p.module_offset, dtype=float)
        hinge = np.asarray(position)+offset
        original = self.original_positions[pid]+offset
        old_depth = (original-self.camera)@self.axis
        if old_depth <= .001:
            raise ValueError(f'LightBender {pid} is behind or too close to the design camera')
        scale = ((hinge-self.camera)@self.axis)/old_depth
        lengths = p.rod_lengths or [p.max_length_limit]*2
        intervals = p.lit_intervals
        if intervals is None:
            intervals = [[[0., p.length_1]], [[0., p.length_2]]]
        lit, dark = [], []
        for rod in range(2):
            if p.rod_directions is None:
                angle, yaw = np.radians([p.angle_1 if rod == 0 else p.angle_2, p.yaw])
                direction = np.array([-np.sin(yaw)*np.cos(angle),
                                      np.cos(yaw)*np.cos(angle), -np.sin(angle)])
            else:
                direction = np.asarray(p.rod_directions[rod])
            on = merge_intervals([(a*scale, b*scale) for a,b in intervals[rod]], lengths[rod])
            off, end = [], 0.
            for a,b in on:
                if a > end:
                    off.append((end, a))
                end = b
            if end < lengths[rod]:
                off.append((end, lengths[rod]))
            colored = [(max(0.,a*scale),min(lengths[rod],b*scale),
                        p.lit_colors[rod][j] if p.lit_colors is not None else None)
                       for j,(a,b) in enumerate(intervals[rod]) if b > a+1e-10]
            for ranges, dest, radius in [(colored, lit, 0.), ([(a,b,None) for a,b in off], dark, p.rod_radius)]:
                for a,b,color in ranges:
                    if b <= a+1e-10:
                        continue
                    world = hinge+np.array([a,b])[:,None]*direction-self.camera
                    depth = world@self.axis
                    if np.min(depth) <= .001:
                        raise ValueError(f'Rod of LightBender {pid} reaches the camera plane')
                    uv = (world@self.basis.T)/depth[:,None]
                    dest.append(Segment(pid, rod+1, uv, depth, radius/np.min(depth), a, b, color))
        return lit, dark

    def update(self, pid, position):
        self.entries[pid] = self.geometry(pid, position)
        self.dirty = True

    def _arrays(self):
        if self.dirty:
            self.flat = [[s for entry in self.entries.values() for s in entry[k]] for k in (0,1)]
            self.bounds = [np.array([s.bounds for s in group]).reshape(-1,4) for group in self.flat]
            self.owners = [np.array([s.owner for s in group]) for group in self.flat]
            self.dirty = False

    def segment_conflicts(self, pid, position=None, against=None, foreground_only=False):
        self._arrays()
        lit, dark = self.entries[pid] if position is None else self.geometry(pid, position)
        groups = [(dark,0)] if foreground_only else [(dark,0),(lit,1)]
        for candidates, other_kind in groups:
            bounds, owners = self.bounds[other_kind], self.owners[other_kind]
            for segment in candidates:
                box = segment.bounds
                mask = ((owners != pid) & (bounds[:,0] <= box[2]) & (bounds[:,2] >= box[0])
                        & (bounds[:,1] <= box[3]) & (bounds[:,3] >= box[1]))
                if against is not None:
                    mask &= np.isin(owners, list(against))
                for index in np.flatnonzero(mask):
                    other = self.flat[other_kind][index]
                    foreground, background = (segment,other) if other_kind == 0 else (other,segment)
                    if blocks(foreground, background):
                        yield foreground, background

    def conflicts(self, pid, position=None, against=None, repairable=True):
        for foreground, background in self.segment_conflicts(pid, position, against):
            if not repairable and collinear_range(foreground,background) is not None:
                continue
            yield foreground.owner, foreground.rod, background.owner, background.rod

    def all_conflicts(self, repairable=True):
        return {conflict for pid in self.entries for conflict in self.conflicts(pid, repairable=repairable)}

    def repaired_lighting(self):
        """Transfer dark projected overlaps, nearest source first; retain other colors.

        Collinear repairs project onto the existing lit interval exactly.
        Crossings cover the finite rod-width footprint (LED sampling limits
        their fidelity). Other viewing angles can change after repair.
        Values are final physical distances, not original unscaled lengths.
        """
        additions = {}
        for pid in self.entries:
            for dark,lit in self.segment_conflicts(pid,foreground_only=True):
                def physical(s):
                    alpha = (s/dark.depth[1])/((1-s)/dark.depth[0]+s/dark.depth[1])
                    return float(dark.start+alpha*(dark.end-dark.start))
                overlap = collinear_range(dark,lit)
                ranges = [overlap] if overlap is not None else _capsule_intervals(
                    *dark.uv, *lit.uv, dark.radius)
                for lo,hi in ranges:
                    for lo,hi in front_ranges(dark,lit,lo,hi) if hi > lo+1e-10 else []:
                        start,end = physical(lo),physical(hi)
                        if end <= start+1e-10:
                            continue
                        # Inverse source depth multiplied by foreground depth is
                        # affine in physical rod distance. This lets us resolve
                        # competing colors exactly where tilted sources swap order.
                        v = lit.uv[1]-lit.uv[0]
                        vv = v@v
                        def depth_numerator(t):
                            uv = dark.uv[0]+t*(dark.uv[1]-dark.uv[0])
                            q = np.clip((uv-lit.uv[0])@v/vv,0.,1.) if vv > 1e-24 else 0.
                            inv_source = (1-q)/lit.depth[0]+q/lit.depth[1]
                            inv_front = (1-t)/dark.depth[0]+t/dark.depth[1]
                            return inv_source/inv_front
                        n0,n1 = depth_numerator(lo),depth_numerator(hi)
                        slope = (n1-n0)/(end-start)
                        additions.setdefault((dark.owner,dark.rod),[]).append(
                            (start,end,lit.color,n0-slope*start,slope,lit,dark))
        # Resolve colors on each foreground rod before assigning ownership.
        proposed = []
        for pid,rod in additions:
            added = additions[pid,rod]
            edges = {x for span in added for x in span[:2]}
            for j,span in enumerate(added):
                a,b,_,c,m = span[:5]
                for other in added[:j]:
                    aa,bb,_,cc,mm = other[:5]
                    if abs(m-mm) > 1e-12:
                        crossing = (cc-c)/(m-mm)
                        if max(a,aa) < crossing < min(b,bb):
                            edges.add(crossing)
            edges = sorted(edges)
            for a,b in zip(edges,edges[1:]):
                if b <= a+1e-10:
                    continue
                mid = (a+b)*.5
                candidates = [span for span in added if span[0] <= mid <= span[1]]
                if candidates:
                    winner = max(candidates,key=lambda span:span[3]+span[4]*mid)
                    proposed.append((slice_segment(winner[6],a,b,winner[2]),winner[5]))
        if not proposed:
            return {}

        original = [segment for lit,_ in self.entries.values() for segment in lit]
        occluders = [replace(segment,radius=self.points[segment.owner].rod_radius/min(segment.depth))
                     for segment in original+[patch for patch,_ in proposed]]
        bounds = np.array([segment.bounds for segment in occluders]).reshape(-1,4)
        visible = []
        for patch,source in proposed:
            # If several dark rods cover one source, only the frontmost receives
            # light. Otherwise a transfer would duplicate it on every layer.
            hidden = list(covered_intervals(patch,occluders,bounds))
            for a,b in subtract_intervals(patch.start,patch.end,hidden):
                visible.append((slice_segment(patch,a,b,patch.color),source))

        removed = {}
        for patch,source in visible:
            coverage = list(covered_intervals(source,[patch],np.array([patch.bounds])))
            if coverage:
                removed.setdefault(id(source),[]).extend(coverage)

        affected = {p.owner for p,_ in visible} | {s.owner for s in original if id(s) in removed}
        result = {pid:{'lit_intervals':[[],[]],'lit_colors':[[],[]],'occlusion_patches':[]}
                  for pid in affected}
        final = {pid:[] for pid in affected}
        for segment in original:
            if segment.owner not in affected:
                continue
            cuts = merge_intervals(removed.get(id(segment),[]),segment.end)
            for a,b in subtract_intervals(segment.start,segment.end,cuts):
                final[segment.owner].append((segment.rod,a,b,segment.color))
            for a,b in cuts:
                result[segment.owner]['occlusion_patches'].append(
                    {'rod':segment.rod,'start':a,'end':b,'color':[0,0,0]})
        for patch,_ in visible:
            final[patch.owner].append((patch.rod,patch.start,patch.end,patch.color))
            result[patch.owner]['occlusion_patches'].append(
                {'rod':patch.rod,'start':patch.start,'end':patch.end,'color':patch.color})
        for pid,spans in final.items():
            for rod in (1,2):
                merged = []
                for _,a,b,color in sorted((v for v in spans if v[0]==rod),key=lambda v:v[1]):
                    if merged and abs(a-merged[-1][1])<1e-10 and color==merged[-1][2]:
                        merged[-1][1] = b
                    else:
                        merged.append([a,b,color])
                result[pid]['lit_intervals'][rod-1] = [[float(a),float(b)] for a,b,_ in merged]
                result[pid]['lit_colors'][rod-1] = [color for _,_,color in merged]
            patches = sorted(result[pid]['occlusion_patches'],key=lambda v:(v['rod'],v['start']))
            merged = []
            for patch in patches:
                if (merged and merged[-1]['rod']==patch['rod'] and merged[-1]['color']==patch['color']
                        and abs(merged[-1]['end']-patch['start'])<1e-10):
                    merged[-1]['end'] = patch['end']
                else:
                    merged.append(patch)
            result[pid]['occlusion_patches'] = [dict(v,start=float(v['start']),end=float(v['end'])) for v in merged]
        return result
