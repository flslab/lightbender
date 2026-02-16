import numpy as np
import yaml
import math
import os
import xml.etree.ElementTree as ET


class PerspectiveCamera:
    def __init__(self, position, target, fov_deg=60, width=800, height=600):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.width = width
        self.height = height
        self.fov_deg = fov_deg

        # Intrinsic Parameters
        # Assuming square pixels and center principal point
        # fov = 2 * arctan(h / (2*f)) => f = h / (2 * tan(fov/2))
        self.f = (height / 2) / math.tan(math.radians(fov_deg / 2))
        self.cx = width / 2
        self.cy = height / 2

        self.view_matrix = self._compute_view_matrix()

    def _compute_view_matrix(self):
        """Computes World-to-Camera View Matrix (LookAt)."""
        up = np.array([0, 0, 1], dtype=float)  # Global Z is up

        f = (self.target - self.position)
        dist = np.linalg.norm(f)
        if dist == 0:
            return np.eye(4)
        f = f / dist

        # Check if f is parallel to up
        if abs(np.dot(f, up)) > 0.99:
            up = np.array([0, 1, 0], dtype=float)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        # Standard CV Convention: Z is forward, X is right, Y is down
        z_axis = f
        x_axis = s
        y_axis = np.cross(z_axis, x_axis)

        R = np.array([x_axis, y_axis, z_axis])
        t = -R @ self.position

        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = t
        return view

    def project_point(self, point_3d):
        """
        Projects a 3D point (x,y,z) to 2D image coordinates (u,v).
        Returns None if point is behind camera.
        """
        p_h = np.append(point_3d, 1.0)
        p_cam = self.view_matrix @ p_h

        x_c, y_c, z_c = p_cam[:3]

        # Check if behind camera (assuming Z+ is forward)
        if z_c <= 0.1:
            return None

        # Pinhole projection
        u = (x_c * self.f) / z_c + self.cx
        v = (y_c * self.f) / z_c + self.cy

        return (u, v)


class SVGWriter:
    def __init__(self, filename, width, height):
        self.filename = filename
        self.width = width
        self.height = height
        self.elements = []

    def add_line(self, x1, y1, x2, y2, color, stroke_width=2, opacity=1.0, element_id=None):
        # Insert ID attribute if provided
        id_attr = f'id="{element_id}" ' if element_id else ''
        self.elements.append(
            f'<line {id_attr}x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-opacity="{opacity}" />'
        )

    def add_circle(self, cx, cy, r, color, fill_opacity=1.0):
        self.elements.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" '
            f'fill="{color}" fill-opacity="{fill_opacity}" />'
        )

    def add_text(self, x, y, text, color="black", size=12):
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-family="Arial" font-size="{size}">{text}</text>'
        )

    def save(self):
        header = f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg" style="background-color:white">'
        footer = '</svg>'
        with open(self.filename, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(self.elements))
            f.write('\n' + footer)
        print(f"SVG saved to {os.path.abspath(self.filename)}")


def get_line_tip_geometry(cx, cy, cz, length, angle_deg):
    """
    Reconstructs the 3D tip of the line based on Body Frame rules.
    Rule: 0 deg = +Y, Axis = -X.
    """
    rad = np.radians(angle_deg)
    dy = length * np.cos(rad)
    dz = -length * np.sin(rad)
    dx = 0
    return np.array([cx + dx, cy + dy, cz + dz])


def render_scene(yaml_input, svg_output, camera_pos):
    # 1. Load Data
    if not os.path.exists(yaml_input):
        print(f"Error: {yaml_input} not found.")
        return

    with open(yaml_input, 'r') as f:
        data = yaml.safe_load(f)

    points_data = data.get('points', [])
    if not points_data:
        print("No points found in YAML.")
        return

    # 2. Setup Camera
    positions = np.array([[p['x'], p['y'], p['z']] for p in points_data])
    centroid = np.mean(positions, axis=0)

    WIDTH, HEIGHT = 1000, 800
    cam = PerspectiveCamera(camera_pos, centroid, fov_deg=60, width=WIDTH, height=HEIGHT)

    # 3. Setup SVG
    svg = SVGWriter(svg_output, WIDTH, HEIGHT)

    # 4. Render
    points_with_dist = []
    for p in points_data:
        pos = np.array([p['x'], p['y'], p['z']])
        dist = np.linalg.norm(pos - np.array(camera_pos))
        points_with_dist.append((dist, p))

    points_with_dist.sort(key=lambda x: x[0], reverse=True)

    for _, p in points_with_dist:
        pid = p['id']
        origin = np.array([p['x'], p['y'], p['z']])

        proj_origin = cam.project_point(origin)
        if proj_origin is None:
            continue

        # Line 1
        tip1_3d = get_line_tip_geometry(origin[0], origin[1], origin[2], p['length_1'], p['angle_1'])
        proj_tip1 = cam.project_point(tip1_3d)

        if proj_tip1:
            svg.add_line(proj_origin[0], proj_origin[1], proj_tip1[0], proj_tip1[1],
                         color="green", stroke_width=3, opacity=0.8,
                         element_id=f"p{pid}_l1")

        # Line 2
        tip2_3d = get_line_tip_geometry(origin[0], origin[1], origin[2], p['length_2'], p['angle_2'])
        proj_tip2 = cam.project_point(tip2_3d)

        if proj_tip2:
            svg.add_line(proj_origin[0], proj_origin[1], proj_tip2[0], proj_tip2[1],
                         color="magenta", stroke_width=3, opacity=0.8,
                         element_id=f"p{pid}_l2")

        svg.add_circle(proj_origin[0], proj_origin[1], 5, color="orange")
        svg.add_text(proj_origin[0] + 8, proj_origin[1] + 4, str(pid), color="#555")

    svg.save()


def compare_svgs(svg_file_1, svg_file_2):
    """
    Parses two SVG files and reports the similarity (average pixel distance)
    between corresponding line endpoints.
    """

    def parse_lines(filename):
        if not os.path.exists(filename):
            print(f"Comparison File not found: {filename}")
            return {}

        try:
            tree = ET.parse(filename)
        except ET.ParseError as e:
            print(f"Failed to parse XML in {filename}: {e}")
            return {}

        root = tree.getroot()
        # SVG elements are usually in a namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        lines = {}
        # Find all lines in the default namespace
        for line in root.findall('.//svg:line', ns):
            lid = line.get('id')
            if lid:
                try:
                    coords = [float(line.get(attr)) for attr in ['x1', 'y1', 'x2', 'y2']]
                    lines[lid] = coords
                except (ValueError, TypeError):
                    continue
        return lines

    print(f"\n--- Comparing SVG Files: {svg_file_1} vs {svg_file_2} ---")
    set1 = parse_lines(svg_file_1)
    set2 = parse_lines(svg_file_2)

    if not set1 or not set2:
        print("Comparison aborted: Could not extract lines from one or both files.")
        return

    common_ids = set(set1.keys()) & set(set2.keys())
    if not common_ids:
        print("No matching line IDs found. Ensure SVGs were generated with IDs.")
        return

    total_diff = 0.0
    num_endpoints = 0

    for lid in common_ids:
        # l1 format: [x1, y1, x2, y2]
        l1 = set1[lid]
        l2 = set2[lid]

        # Distance between Start Points
        d_start = math.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2)
        # Distance between End Points
        d_end = math.sqrt((l1[2] - l2[2]) ** 2 + (l1[3] - l2[3]) ** 2)

        total_diff += (d_start + d_end)
        num_endpoints += 2

    avg_diff = total_diff / num_endpoints if num_endpoints > 0 else 0.0

    print(f"Matched Lines: {len(common_ids)}")
    print(f"Average Endpoint Difference: {avg_diff:.4f} pixels")
    print(f"Similarity Score (1000 / (1000 + avg_diff)): {1000.0 / (1000.0 + avg_diff):.4f}")
    print("---------------------------------------------------")


if __name__ == "__main__":
    # Configuration
    CAMERA_POSITION = (2.3, 0.0, 0.8)

    FILE_BEFORE_YAML = "points_input.yaml"
    FILE_AFTER_YAML = "points_output.yaml"

    SVG_BEFORE = "before.svg"
    SVG_AFTER = "after.svg"

    # 1. Render 'Before' state if available
    if os.path.exists(FILE_BEFORE_YAML):
        print(f"Rendering Before State: {FILE_BEFORE_YAML}")
        render_scene(FILE_BEFORE_YAML, SVG_BEFORE, CAMERA_POSITION)

    # 2. Render 'After' state if available
    if os.path.exists(FILE_AFTER_YAML):
        print(f"Rendering After State: {FILE_AFTER_YAML}")
        render_scene(FILE_AFTER_YAML, SVG_AFTER, CAMERA_POSITION)

    # 3. Compare if both SVGs exist
    if os.path.exists(SVG_BEFORE) and os.path.exists(SVG_AFTER):
        compare_svgs(SVG_BEFORE, SVG_AFTER)