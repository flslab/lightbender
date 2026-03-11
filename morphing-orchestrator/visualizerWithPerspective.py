import yaml
import os
import sys
import math
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "./mission_visualizations"
LINE_LENGTH_CM = 31.2  # 312 mm = 31.2 cm

DEFAULT_CAMERA_POS = (235.0, 0.0, 1.0)  # Sensible default in cm-space


# -----------------------------
# Utility Functions
# -----------------------------
def meters_to_cm(value):
    return value * 100.0


# =====================================================================
# Perspective Camera (from visualizerWithPerspective.py)
# =====================================================================
class PerspectiveCamera:
    def __init__(self, position, target, fov_deg=60, width=1920, height=1080):
        self.position = np.array(position, dtype=float)
        self.target   = np.array(target, dtype=float)
        self.width    = width
        self.height   = height
        self.fov_deg  = fov_deg

        # Intrinsic parameters
        self.f  = (height / 2) / math.tan(math.radians(fov_deg / 2))
        self.cx = width  / 2
        self.cy = height / 2

        self.view_matrix = self._compute_view_matrix()

    def _compute_view_matrix(self):
        """World-to-Camera LookAt matrix (Z-forward, X-right, Y-down)."""
        up = np.array([0, 0, 1], dtype=float)  # Global Z is up

        f = self.target - self.position
        dist = np.linalg.norm(f)
        if dist == 0:
            return np.eye(4)
        f = f / dist

        if abs(np.dot(f, up)) > 0.99:
            up = np.array([0, 1, 0], dtype=float)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        z_axis = f
        x_axis = s
        y_axis = np.cross(z_axis, x_axis)

        R = np.array([x_axis, y_axis, z_axis])
        t = -R @ self.position

        view = np.eye(4)
        view[:3, :3] = R
        view[:3,  3] = t
        return view

    def project_point(self, point_3d):
        """
        Project a 3D world point → 2D pixel (u, v).
        Returns None if the point is behind the camera.
        """
        p_h   = np.append(np.array(point_3d, dtype=float), 1.0)
        p_cam = self.view_matrix @ p_h
        x_c, y_c, z_c = p_cam[:3]

        if z_c <= 0.1:
            return None

        u = (x_c * self.f) / z_c + self.cx
        v = (y_c * self.f) / z_c + self.cy
        return (u, v)

    def depth(self, point_3d):
        """Camera-space Z depth of a world point (for painter's sort)."""
        p_h   = np.append(np.array(point_3d, dtype=float), 1.0)
        p_cam = self.view_matrix @ p_h
        return p_cam[2]


# =====================================================================
# SVG Writer (from visualizerWithPerspective.py, extended)
# =====================================================================
class SVGWriter:
    def __init__(self, filename, width, height):
        self.filename = filename
        self.width    = width
        self.height   = height
        self.elements = []

    def add_line(self, x1, y1, x2, y2, color="black", stroke_width=2,
                 opacity=1.0, element_id=None):
        id_attr = f'id="{element_id}" ' if element_id else ''
        self.elements.append(
            f'<line {id_attr}x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{stroke_width:.2f}" '
            f'stroke-opacity="{opacity}" />'
        )

    def add_circle(self, cx, cy, r, color="orange", fill_opacity=1.0):
        self.elements.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" '
            f'fill="{color}" fill-opacity="{fill_opacity}" />'
        )

    def add_text(self, x, y, text, color="black", size=14, bold=False):
        weight = "bold" if bold else "normal"
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" '
            f'font-family="Arial" font-size="{size}" font-weight="{weight}">'
            f'{text}</text>'
        )

    def add_arrow(self, x1, y1, x2, y2, color="black",
                  stroke_width=2, head_size=10):
        """
        Draw a line with an arrowhead at (x2, y2).
        Uses an SVG marker for a clean arrowhead.
        """
        marker_id = f"arrow_{len(self.elements)}"
        # Define marker in defs (just inline each time — small overhead)
        defs = (
            f'<defs>'
            f'<marker id="{marker_id}" markerWidth="6" markerHeight="6" '
            f'refX="3" refY="3" orient="auto">'
            f'<path d="M0,0 L0,6 L6,3 z" fill="{color}" />'
            f'</marker>'
            f'</defs>'
        )
        self.elements.append(defs)
        self.elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{stroke_width:.2f}" '
            f'marker-end="url(#{marker_id})" />'
        )

    def add_grid(self, spacing=100, color="#e0e0e0"):
        """Draw a light reference grid."""
        for x in range(0, self.width, spacing):
            self.elements.append(
                f'<line x1="{x}" y1="0" x2="{x}" y2="{self.height}" '
                f'stroke="{color}" stroke-width="1" />'
            )
        for y in range(0, self.height, spacing):
            self.elements.append(
                f'<line x1="0" y1="{y}" x2="{self.width}" y2="{y}" '
                f'stroke="{color}" stroke-width="1" />'
            )

    def add_legend(self, entries, x=20, y=30):
        """entries: list of (color, label) tuples."""
        for i, (color, label) in enumerate(entries):
            row_y = y + i * 22
            self.elements.append(
                f'<rect x="{x}" y="{row_y - 12}" width="16" height="14" '
                f'fill="{color}" />'
            )
            self.elements.append(
                f'<text x="{x + 22}" y="{row_y}" fill="black" '
                f'font-family="Arial" font-size="13">{label}</text>'
            )

    def save(self):
        header = (
            f'<svg width="{self.width}" height="{self.height}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:white">'
        )
        footer = '</svg>'
        with open(self.filename, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(self.elements))
            f.write('\n' + footer)


# =====================================================================
# Camera-projected Matplotlib Helpers
# =====================================================================
def draw_lightbender_projected(ax, cam, center_cm, length_cm, label, color="C0"):
    """
    Project the lightbender's left/mid/right endpoints through the camera and draw
    the resulting 2D line in matplotlib pixel-space.
    """
    left, mid_px, right = project_lightbender(cam, center_cm, length_cm)

    if left and right:
        depth = cam.depth(center_cm)
        lw = max(1.0, 6.0 / max(depth, 1.0) * 200)
        ax.plot([left[0], right[0]], [left[1], right[1]],
                linewidth=lw, color=color)

    if mid_px:
        ax.scatter(mid_px[0], mid_px[1], s=50, color=color, zorder=5)
        ax.text(mid_px[0] + 8, mid_px[1] - 8, label,
                color=color, fontsize=9, fontweight="bold")


def draw_direction_arrow_projected(ax, cam, start_3d, end_3d,
                                   color="black", min_movement=5):
    """
    Project start/end through the camera then draw a 2D arrow in pixel-space.
    """
    p_start = cam.project_point(start_3d)
    p_end   = cam.project_point(end_3d)

    if p_start is None or p_end is None:
        return

    s = np.array(p_start)
    e = np.array(p_end)
    if np.linalg.norm(e - s) < min_movement:
        return

    arrow = FancyArrowPatch(
        (s[0], s[1]), (e[0], e[1]),
        arrowstyle='-|>', mutation_scale=12, linewidth=2, color=color
    )
    ax.add_patch(arrow)


# =====================================================================
# Perspective Rendering Helpers
# =====================================================================
def project_lightbender(cam, center_cm, length_cm):
    """
    Return projected (left, mid, right) pixel coords for a horizontal lightbender,
    or (None, None, None) if occluded.
    Offset is along the world X axis so the bar appears horizontal from any POV.
    """
    x, y, z = center_cm
    half  = length_cm / 2.0
    left  = cam.project_point([x - half, y, z])
    mid   = cam.project_point([x,        y, z])
    right = cam.project_point([x + half, y, z])
    return left, mid, right


def render_perspective_frame(cam, svg, lb3_cm, lb2_cm, next_lb3_cm, next_lb2_cm,
                              length_cm, frame_idx):
    """
    Draw one frame's worth of scene elements into the SVG using the camera.
    Draws back-to-front (painter's algorithm) based on depth.
    """

    # ---- Collect drawable objects with depth ----
    objects = []

    # Lightbenders
    for center, label, color in [(lb3_cm, "FLS E", "#2980b9"),
                                  (lb2_cm, "FLS F", "#e74c3c")]:
        depth = cam.depth(center)
        objects.append(("lightbender", depth, center, label, color))

    # Motion arrows (depth = midpoint)
    if next_lb3_cm is not None:
        mid = [(a + b) / 2 for a, b in zip(lb3_cm, next_lb3_cm)]
        objects.append(("arrow", cam.depth(mid), lb3_cm, next_lb3_cm, "#2c3e50"))

    if next_lb2_cm is not None:
        mid = [(a + b) / 2 for a, b in zip(lb2_cm, next_lb2_cm)]
        objects.append(("arrow", cam.depth(mid), lb2_cm, next_lb2_cm, "#7f8c8d"))

    # Sort back → front (largest depth first)
    objects.sort(key=lambda o: o[1], reverse=True)

    # ---- Draw ----
    for obj in objects:
        kind = obj[0]

        if kind == "lightbender":
            _, depth, center, label, color = obj
            left, mid_px, right = project_lightbender(cam, center, length_cm)
            if left and right:
                # Approximate projected stroke width (thicker when closer)
                stroke = max(1.5, 8.0 / max(depth, 1.0) * 200)
                svg.add_line(left[0], left[1], right[0], right[1],
                             color=color, stroke_width=stroke, opacity=0.9)
            if mid_px:
                svg.add_circle(mid_px[0], mid_px[1], 6, color=color)
                svg.add_text(mid_px[0] + 10, mid_px[1] - 8, label,
                             color=color, size=15, bold=True)

        elif kind == "arrow":
            _, depth, start_3d, end_3d, color = obj
            p_start = cam.project_point(start_3d)
            p_end   = cam.project_point(end_3d)
            if p_start and p_end:
                svg.add_arrow(p_start[0], p_start[1],
                              p_end[0],   p_end[1],
                              color=color, stroke_width=2)


# =====================================================================
# Main
# =====================================================================
def main(mission_file, use_perspective=False, camera_pos=DEFAULT_CAMERA_POS,
         svg_width=1920, svg_height=1080):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(mission_file, "r") as f:
        mission = yaml.safe_load(f)

    lb3_waypoints = mission["drones"]["lb3"]["waypoints"]
    lb2_waypoints = mission["drones"]["lb2"]["waypoints"]

    number_of_images = max(len(lb3_waypoints), len(lb2_waypoints)) - 1
    print(f"Generating {number_of_images} frame(s)...")

    # Pre-compute scene centroid for camera targeting
    all_pts = []
    for wp in lb3_waypoints + lb2_waypoints:
        all_pts.append([meters_to_cm(v) for v in wp[:3]])
    scene_centroid = np.mean(all_pts, axis=0) if all_pts else [0, 0, 0]

    # Build perspective camera once — used by BOTH modes
    cam = PerspectiveCamera(
        position=camera_pos,
        target=scene_centroid,
        fov_deg=60,
        width=svg_width,
        height=svg_height
    )

    for i in range(number_of_images):
        # Current waypoints → cm
        lb3_cm = ([meters_to_cm(v) for v in lb3_waypoints[i][:3]]
                  if i < len(lb3_waypoints) else None)
        lb2_cm = ([meters_to_cm(v) for v in lb2_waypoints[i][:3]]
                  if i < len(lb2_waypoints) else None)

        # Next waypoints (for arrows)
        next_lb3_cm = ([meters_to_cm(v) for v in lb3_waypoints[i + 1][:3]]
                       if i < len(lb3_waypoints) - 1 else None)
        next_lb2_cm = ([meters_to_cm(v) for v in lb2_waypoints[i + 1][:3]]
                       if i < len(lb2_waypoints) - 1 else None)

        # ------------------------------------------------------------------
        # PERSPECTIVE MODE  →  SVG output
        # ------------------------------------------------------------------
        if use_perspective:
            output_path = os.path.join(OUTPUT_DIR, f"mission_frame_{i + 1}.svg")
            svg = SVGWriter(output_path, svg_width, svg_height)

            # Background grid for spatial reference
            svg.add_grid(spacing=80, color="#f0f0f0")

            # Frame title
            svg.add_text(svg_width / 2 - 120, 30,
                         f"Perspective View — Frame {i + 1}",
                         color="#333", size=18, bold=True)

            # Camera info annotation
            cp = camera_pos
            svg.add_text(10, svg_height - 10,
                         f"Camera @ ({cp[0]:.0f}, {cp[1]:.0f}, {cp[2]:.0f}) cm  "
                         f"| Target: scene centroid  | FOV 60°",
                         color="#888", size=11)

            # Legend
            svg.add_legend([("#2980b9", "FLS E (lb3)"),
                            ("#e74c3c", "FLS F (lb2)"),
                            ("#2c3e50", "Motion arrow — lb3"),
                            ("#7f8c8d", "Motion arrow — lb2")],
                           x=svg_width - 220, y=30)

            # Render scene objects
            if lb3_cm and lb2_cm:
                render_perspective_frame(cam, svg,
                                         lb3_cm, lb2_cm,
                                         next_lb3_cm, next_lb2_cm,
                                         LINE_LENGTH_CM, i)

            svg.save()

        # ------------------------------------------------------------------
        # 2D MATPLOTLIB MODE — projected through the perspective camera
        # ------------------------------------------------------------------
        else:
            fig, ax = plt.subplots(figsize=(svg_width / 100, svg_height / 100))

            # Motion arrows (draw first so lightbenders render on top)
            if lb3_cm and next_lb3_cm:
                draw_direction_arrow_projected(ax, cam, lb3_cm, next_lb3_cm,
                                               color="#2c3e50")
            if lb2_cm and next_lb2_cm:
                draw_direction_arrow_projected(ax, cam, lb2_cm, next_lb2_cm,
                                               color="#7f8c8d")

            # Lightbenders
            if lb3_cm:
                draw_lightbender_projected(ax, cam, lb3_cm, LINE_LENGTH_CM,
                                           "FLS E", color="#2980b9")
            if lb2_cm:
                draw_lightbender_projected(ax, cam, lb2_cm, LINE_LENGTH_CM,
                                           "FLS F", color="#e74c3c")

            # Axes are now in pixel-space matching the camera's image plane
            ax.set_xlim(0, svg_width)
            ax.set_ylim(svg_height, 0)   # flip Y so image-top is up
            ax.set_xlabel("u (px)")
            ax.set_ylabel("v (px)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#e0e0e0")

            cp = camera_pos
            ax.set_title(
                f"Camera POV — Frame {i + 1}  |  "
                f"Camera @ ({cp[0]:.0f}, {cp[1]:.0f}, {cp[2]:.0f}) cm  |  FOV 60°"
            )

            output_path = os.path.join(OUTPUT_DIR, f"mission_frame_{i + 1}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close(fig)

        print(f"  Saved: {output_path}")

    print("Done.")


# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mission Visualizer — 2D top-down or Perspective camera SVG output"
    )
    parser.add_argument("mission", help="Path to mission YAML file")
    parser.add_argument(
        "--perspective", action="store_true",
        help="Render frames using perspective camera (SVG output). "
             "Default: 2D top-down matplotlib PNG."
    )
    parser.add_argument(
        "--camera_pos", type=float, nargs=3,
        default=list(DEFAULT_CAMERA_POS),
        metavar=("X", "Y", "Z"),
        help="Camera position in cm (default: 200 -300 150)"
    )
    parser.add_argument(
        "--width",  type=int, default=1920, help="SVG output width  (default 1920)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="SVG output height (default 1080)"
    )

    args = parser.parse_args()

    main(
        mission_file=args.mission,
        use_perspective=args.perspective,
        camera_pos=tuple(args.camera_pos),
        svg_width=args.width,
        svg_height=args.height,
    )