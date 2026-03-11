import yaml
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "./mission_visualizations"
LINE_LENGTH_CM = 31.2  # 312 mm = 31.2 cm


# -----------------------------
# Utility Functions
# -----------------------------
def meters_to_cm(value):
    return value * 100.0


def draw_lightbender(ax, center_cm, length_cm, label):
    """
    Draw a vertical 2D line centered at (X, Y).
    Z is ignored in 2D view.
    """
    x, y, _ = center_cm
    half = length_cm / 2.0

    y_start = y - half
    y_end = y + half

    ax.plot([x, x], [y_start, y_end], linewidth=3)
    ax.scatter(x, y, s=50)
    ax.text(x, y, label)

def draw_lightbender_3d(ax, center_cm, length_cm, label):
    """
    Draw vertical 3D lightbender centered at (X, Y, Z)
    """
    x, y, z = center_cm
    half = length_cm / 2.0

    z_start = z - half
    z_end = z + half

    ax.plot([x, x], [y, y], [z_start, z_end], linewidth=3)
    ax.scatter(x, y, z, s=50)
    ax.text(x, y, z, label)

def draw_direction_arrow(ax, start, end,
                         offset_distance=2.0,
                         min_movement=20):
    """
    Draw arrow exactly from start → end.
    No artificial extension.
    Handles tiny movements cleanly.
    """

    start = np.array(start[:2], dtype=float)
    end = np.array(end[:2], dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)
    print(length)

    # Draw the tiny motion
    if length < min_movement:
        arrow = FancyArrowPatch(
            (start[0], start[1]),
            (end[0], end[1]),
            arrowstyle='-|>',
            mutation_scale=12,
            linewidth=2,
            color="black"
        )

        ax.add_patch(arrow)
        return

    # Unit direction
    unit_dir = direction / length

    # Perpendicular offset (small, consistent)
    perp = np.array([-unit_dir[1], unit_dir[0]])
    offset = perp * offset_distance

    start_offset = start + offset
    end_offset = end + offset

    # Scale arrow head relative to movement
    mutation = max(6, min(15, 0.4 * length))

    arrow = FancyArrowPatch(
        (start_offset[0], start_offset[1]),
        (end_offset[0], end_offset[1]),
        arrowstyle='-|>',
        mutation_scale=mutation,
        linewidth=2,
        color="black"
    )

    ax.add_patch(arrow)

def draw_direction_arrow_3d(ax, start, end, min_movement=2.0):
    """
    Draw true 3D arrow from start to end
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)

    if length < min_movement:
        return

    ax.quiver(
        start[0], start[1], start[2],
        direction[0], direction[1], direction[2],
        arrow_length_ratio=0.15,
        linewidth=2,
        color="black"
    )

# -----------------------------
# Main Script
# -----------------------------
def main(mission_file, use_3d=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(mission_file, "r") as f:
        mission = yaml.safe_load(f)

    lb3_waypoints = mission["drones"]["lb3"]["waypoints"]
    lb2_waypoints = mission["drones"]["lb2"]["waypoints"]

    number_of_images = max(len(lb3_waypoints), len(lb2_waypoints))

    points_for_scaling = []

    for i in range(number_of_images):
        if use_3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()

        # lb3 current waypoint
        if i < len(lb3_waypoints):
            lb3_wp = lb3_waypoints[i][:3]
            lb3_cm = [meters_to_cm(v) for v in lb3_wp]
            points_for_scaling.append(lb3_cm)

        # lb2 current waypoint
        if i < len(lb2_waypoints):
            lb2_wp = lb2_waypoints[i][:3]
            lb2_cm = [meters_to_cm(v) for v in lb2_wp]
            points_for_scaling.append(lb2_cm)

        if i < len(lb3_waypoints) - 1:
            next_lb3_wp = lb3_waypoints[i + 1][:3]
            next_lb3_cm = [meters_to_cm(v) for v in next_lb3_wp]
            if use_3d:
                draw_direction_arrow_3d(ax, lb3_cm, next_lb3_cm)
            else:
                draw_direction_arrow(ax, lb3_cm, next_lb3_cm)


        if i < len(lb2_waypoints) - 1:
            next_lb2_wp = lb2_waypoints[i + 1][:3]
            next_lb2_cm = [meters_to_cm(v) for v in next_lb2_wp]
            if use_3d:
                draw_direction_arrow_3d(ax, lb2_cm, next_lb2_cm)
            else:
                draw_direction_arrow(ax, lb2_cm, next_lb2_cm)

        # Draw lightbenders
        if use_3d:
            draw_lightbender_3d(ax, lb3_cm, LINE_LENGTH_CM, "FLS E")
            draw_lightbender_3d(ax, lb2_cm, LINE_LENGTH_CM, "FLS F")
        else:
            draw_lightbender(ax, lb3_cm, LINE_LENGTH_CM, "FLS E")
            draw_lightbender(ax, lb2_cm, LINE_LENGTH_CM, "FLS F")

        # Axis setup
        if use_3d:
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            ax.set_zlim(-150, 150)
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_zlabel("Z (cm)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
        else:
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)

        # Save figure
        output_path = os.path.join(OUTPUT_DIR, f"mission_frame_{i+1}.png")
        plt.title(f"Mission Visualization (2D) - Frame {i+1}")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    mission = sys.argv[1]
    use_3d = "--3d" in sys.argv
    main(mission, use_3d)