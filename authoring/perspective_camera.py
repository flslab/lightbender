import numpy as np
import yaml
import math
import os
import xml.etree.ElementTree as ET
import argparse
import sys


class PerspectiveCamera:
    def __init__(self, position, target, fov_deg=60, width=800, height=600):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.width = width
        self.height = height
        self.fov_deg = fov_deg

        # Intrinsic Parameters
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
            f'stroke="{color}" stroke-width="{stroke_width:.2f}" stroke-opacity="{opacity}" />'
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


def get_line_tip_geometry(cx, cy, cz, length, angle_deg, yaw_deg):
    """
    Reconstructs the 3D tip of the line based on Body Frame rules + Yaw.
    Rule: Body Frame 0 deg = +Y, Axis = -X.
    Yaw: Rotation about Global +Z.
    """
    rad = np.radians(angle_deg)
    # Local Body Frame
    dy_local = length * np.cos(rad)
    dz_local = -length * np.sin(rad)
    dx_local = 0.0

    # Apply Yaw Rotation (around +Z)
    rad_yaw = np.radians(yaw_deg)
    cos_y = np.cos(rad_yaw)
    sin_y = np.sin(rad_yaw)

    # RotZ * [dx, dy, dz]^T
    dx_rot = dx_local * cos_y - dy_local * sin_y
    dy_rot = dx_local * sin_y + dy_local * cos_y
    dz_rot = dz_local

    return np.array([cx + dx_rot, cy + dy_rot, cz + dz_rot])


def render_scene(yaml_input, svg_output, camera_pos):
    # 1. Load Data
    if not os.path.exists(yaml_input):
        print(f"Error: {yaml_input} not found.", file=sys.stderr)
        return

    with open(yaml_input, 'r') as f:
        data = yaml.safe_load(f)

    points_data = data.get('points', [])
    if not points_data:
        print("No points found in YAML.", file=sys.stderr)
        return

    # 2. Setup Camera
    positions = np.array([[p['x'], p['y'], p['z']] for p in points_data])
    centroid = np.mean(positions, axis=0)

    WIDTH, HEIGHT = 1920, 1080
    cam = PerspectiveCamera(camera_pos, centroid, fov_deg=60, width=WIDTH, height=HEIGHT)

    # 3. Setup SVG
    svg = SVGWriter(svg_output, WIDTH, HEIGHT)

    # Base stroke width for lines
    BASE_STROKE_WIDTH = 3.0

    # 4. Render
    # Sort points by distance to camera
    points_with_dist = []
    for p in points_data:
        pos = np.array([p['x'], p['y'], p['z']])
        dist = np.linalg.norm(pos - np.array(camera_pos))
        points_with_dist.append((dist, p))

    points_with_dist.sort(key=lambda x: x[0], reverse=True)

    for _, p in points_with_dist:
        pid = p['id']
        origin = np.array([p['x'], p['y'], p['z']])
        yaw = p.get('yaw', 0.0)  # Default yaw is 0

        # Get scale factor (default 1.0 if not present)
        scale_factor = p.get('scale_factor', 1.0)
        current_stroke = BASE_STROKE_WIDTH / scale_factor

        proj_origin = cam.project_point(origin)
        if proj_origin is None:
            continue

        split_info = p.get('split_info')
        l1_id = f"p{pid}_l1" if not split_info else None
        l2_id = f"p{pid}_l2" if not split_info else None

        # Line 1
        tip1_3d = get_line_tip_geometry(origin[0], origin[1], origin[2], p['length_1'], p['angle_1'], yaw)
        proj_tip1 = cam.project_point(tip1_3d)

        if proj_tip1:
            svg.add_line(proj_origin[0], proj_origin[1], proj_tip1[0], proj_tip1[1],
                         color="green", stroke_width=current_stroke, opacity=0.8,
                         element_id=l1_id)

        # Line 2
        tip2_3d = get_line_tip_geometry(origin[0], origin[1], origin[2], p['length_2'], p['angle_2'], yaw)
        proj_tip2 = cam.project_point(tip2_3d)

        if proj_tip2:
            svg.add_line(proj_origin[0], proj_origin[1], proj_tip2[0], proj_tip2[1],
                         color="magenta", stroke_width=current_stroke, opacity=0.8,
                         element_id=l2_id)

        if split_info and proj_tip1 and proj_tip2:
            parent_id = split_info.get('parent_id', pid)
            segment_idx = split_info.get('segment', 1)
            color = "green" if segment_idx == 1 else "magenta"
            svg.add_line(proj_tip1[0], proj_tip1[1], proj_tip2[0], proj_tip2[1],
                         color=color, stroke_width=current_stroke, opacity=0.8,
                         element_id=f"p{parent_id}_l{segment_idx}")

        svg.add_circle(proj_origin[0], proj_origin[1], 5, color="orange")
        svg.add_text(proj_origin[0] + 8, proj_origin[1] + 4, str(pid), color="#555")

    svg.save()


def compare_svgs(svg_file_1, svg_file_2, csv_mode=False):
    """
    Parses two SVG files and reports the similarity based on:
    1. Endpoint distance (pixels)
    2. Stroke width difference (pixels)

    Similarity is computed using Exponential Decay on the normalized error.
    Score = exp(- Sensitivity * NormalizedError)
    """

    def parse_svg_data(filename):
        if not os.path.exists(filename):
            if not csv_mode: print(f"Comparison File not found: {filename}")
            return {}, 0.0

        try:
            tree = ET.parse(filename)
        except ET.ParseError as e:
            if not csv_mode: print(f"Failed to parse XML in {filename}: {e}")
            return {}, 0.0

        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        # Extract dimensions to calculate diagonal
        try:
            w = float(root.get('width', 1920))
            h = float(root.get('height', 1080))
            diag = math.sqrt(w ** 2 + h ** 2)
        except (ValueError, TypeError):
            diag = math.sqrt(1920 ** 2 + 1080 ** 2)  # Default fallback diagonal

        lines = {}
        # Try both namespaced and non-namespaced find
        for line in root.findall('.//svg:line', ns) + root.findall('.//line'):
            lid = line.get('id')
            if lid:
                try:
                    coords = [float(line.get(attr)) for attr in ['x1', 'y1', 'x2', 'y2']]
                    width = float(line.get('stroke-width', 1.0))
                    lines[lid] = {'coords': coords, 'width': width}
                except (ValueError, TypeError):
                    continue
        return lines, diag

    if not csv_mode:
        print(f"\n--- Comparing SVG Files: {svg_file_1} vs {svg_file_2} ---")

    set1, diag1 = parse_svg_data(svg_file_1)
    set2, diag2 = parse_svg_data(svg_file_2)

    if not set1 or not set2:
        if not csv_mode: print("Comparison aborted: Could not extract lines from one or both files.")
        return

    common_ids = set(set1.keys()) & set(set2.keys())
    if not common_ids:
        if not csv_mode: print("No matching line IDs found. Ensure SVGs were generated with IDs.")
        return

    # Use the diagonal from the first file (assuming identical resolution)
    img_diagonal = diag1 if diag1 > 0 else 1.0

    # Compute centroids over all endpoints of common lines and subtract before measuring error
    cx1 = cy1 = cx2 = cy2 = 0.0
    n_pts = len(common_ids) * 2  # 2 endpoints per line
    for lid in common_ids:
        c1 = set1[lid]['coords']  # x1,y1,x2,y2
        c2 = set2[lid]['coords']
        cx1 += c1[0] + c1[2]
        cy1 += c1[1] + c1[3]
        cx2 += c2[0] + c2[2]
        cy2 += c2[1] + c2[3]
    cx1 /= n_pts
    cy1 /= n_pts
    cx2 /= n_pts
    cy2 /= n_pts

    total_pos_diff = 0.0
    total_width_diff = 0.0
    num_lines = 0

    for lid in common_ids:
        l1_data = set1[lid]
        l2_data = set2[lid]

        c1 = l1_data['coords']
        w1 = l1_data['width']

        c2 = l2_data['coords']
        w2 = l2_data['width']

        # Centroid-aligned endpoint distances
        # Option A: align (x1, y1) with (x1, y1)
        d1_start = math.sqrt((c1[0] - cx1 - (c2[0] - cx2)) ** 2 + (c1[1] - cy1 - (c2[1] - cy2)) ** 2)
        d1_end   = math.sqrt((c1[2] - cx1 - (c2[2] - cx2)) ** 2 + (c1[3] - cy1 - (c2[3] - cy2)) ** 2)
        dist1 = d1_start + d1_end
        
        # Option B: align (x1, y1) with (x2, y2)
        d2_start = math.sqrt((c1[0] - cx1 - (c2[2] - cx2)) ** 2 + (c1[1] - cy1 - (c2[3] - cy2)) ** 2)
        d2_end   = math.sqrt((c1[2] - cx1 - (c2[0] - cx2)) ** 2 + (c1[3] - cy1 - (c2[1] - cy2)) ** 2)
        dist2 = d2_start + d2_end

        total_pos_diff += min(dist1, dist2)

        # Absolute difference for width
        d_width = abs(w1 - w2)
        total_width_diff += d_width
        num_lines += 1

    # Metrics
    # Average pixel error per feature
    # Total features per line = 2 endpoints + 1 width check = 3
    avg_pos_diff = total_pos_diff / (num_lines * 2) if num_lines > 0 else 0.0

    # Average width difference per line
    avg_width_diff = total_width_diff / num_lines if num_lines > 0 else 0.0

    # Combined error (avg per feature: 2 endpoints + 1 width = 3)
    overall_avg_pixel_error = (total_pos_diff + total_width_diff) / (num_lines * 3) if num_lines > 0 else 0.0

    # Normalize error relative to image diagonal
    normalized_error = overall_avg_pixel_error / img_diagonal

    # --- Score Calculation (Exponential Decay) ---
    # Sensitivity factor: Determines how fast the score drops.
    # Alpha = 100 means:
    #   Normalized Error = 0.01 (1%)  -> Score ~ 0.36
    #   Normalized Error = 0.001 (0.1%) -> Score ~ 0.90
    sensitivity_alpha = 100.0
    score = math.exp(-sensitivity_alpha * normalized_error)

    if csv_mode:
        # CSV: MatchedLines,ImageDiagonal,AvgPosError,AvgWidthError,OverallAvgError,NormalizedError,SimilarityScore
        print(
            f"{len(common_ids)},{img_diagonal:.2f},{avg_pos_diff:.4f},{avg_width_diff:.4f},{overall_avg_pixel_error:.4f},{normalized_error:.6f},{score:.4f}")
    else:
        print(f"Matched Lines:           {len(common_ids)}")
        print(f"Image Diagonal:          {img_diagonal:.2f} pixels")
        print(f"Avg Endpoint Diff:       {avg_pos_diff:.4f} pixels")
        print(f"Avg Stroke Width Diff:   {avg_width_diff:.4f} pixels")
        print(f"Overall Avg Pixel Error: {overall_avg_pixel_error:.4f} pixels")
        print(f"Normalized Error:        {normalized_error:.6f} (relative to diagonal)")
        print(f"Similarity Score:        {score:.4f} (Exponential Decay)")
        print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perspective Camera & SVG Comparison Tool")

    # Modes
    parser.add_argument("--action", choices=['render', 'compare', 'demo'], default='demo',
                        help="Action to perform: 'render' yaml to svg, 'compare' two svgs, or 'demo' default behavior")

    # Arguments
    parser.add_argument("--input", type=str, help="Input file (YAML for render, SVG 1 for compare)")
    parser.add_argument("--output", type=str, help="Output file (SVG for render, SVG 2 for compare)")
    parser.add_argument("--camera_pos", type=float, nargs=3, default=[2.3, 0.0, 0.8], help="Camera Position x y z")
    parser.add_argument("--csv", action="store_true", help="Output comparison metrics in CSV format")

    args = parser.parse_args()

    CAMERA_POS = tuple(args.camera_pos)

    if args.action == 'render':
        if not args.input or not args.output:
            print("Error: --input (YAML) and --output (SVG) required for render mode.")
        else:
            render_scene(args.input, args.output, CAMERA_POS)

    elif args.action == 'compare':
        if not args.input or not args.output:
            print("Error: --input (SVG 1) and --output (SVG 2) required for compare mode.")
        else:
            compare_svgs(args.input, args.output, csv_mode=args.csv)

    elif args.action == 'demo':
        FILE_BEFORE_YAML = "points_input.yaml"
        FILE_AFTER_YAML = "points_output.yaml"
        SVG_BEFORE = "before.svg"
        SVG_AFTER = "after.svg"

        if os.path.exists(FILE_BEFORE_YAML):
            print(f"Rendering Before State: {FILE_BEFORE_YAML}")
            render_scene(FILE_BEFORE_YAML, SVG_BEFORE, CAMERA_POS)

        if os.path.exists(FILE_AFTER_YAML):
            print(f"Rendering After State: {FILE_AFTER_YAML}")
            render_scene(FILE_AFTER_YAML, SVG_AFTER, CAMERA_POS)

        if os.path.exists(SVG_BEFORE) and os.path.exists(SVG_AFTER):
            compare_svgs(SVG_BEFORE, SVG_AFTER)