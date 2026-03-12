import argparse
import xml.etree.ElementTree as ET
import yaml
from svg.path import parse_path, CubicBezier


def extract_paths_from_svg(svg_file):
    """Parses the SVG file and extracts the 'd' attributes from all <path> elements."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    paths = []

    # Iterate over all elements. Using endswith to ignore XML namespace prefixes (e.g. {http://www.w3.org/2000/svg}path)
    for elem in root.iter():
        if elem.tag.endswith('path'):
            d_attr = elem.attrib.get('d')
            if d_attr:
                paths.append(d_attr)

    return paths


def build_raw_graph(paths, curve_samples=5):
    """Parses path data and builds a graph with unscaled, raw coordinates."""
    raw_nodes = []
    edges = []
    vertex_map = {}

    def get_node_id(svg_x, svg_y):
        # Map 2D SVG to 3D graph (x -> y, y -> z)
        mapped_y = svg_x
        mapped_z = svg_y

        # Round to 3 decimal places to merge vertices that are practically identical
        round_y = round(mapped_y, 3)
        round_z = round(mapped_z, 3)
        coord_key = (round_y, round_z)

        if coord_key not in vertex_map:
            node_id = len(raw_nodes)
            vertex_map[coord_key] = node_id
            # Store the exact floating point for accurate later scaling
            raw_nodes.append({'y': mapped_y, 'z': mapped_z})

        return vertex_map[coord_key]

    for path_data in paths:
        parsed_path = parse_path(path_data)

        for segment in parsed_path:
            # Determine number of segments depending on whether it's a curve or line
            num_samples = curve_samples if isinstance(segment, CubicBezier) else 1

            # Start point
            prev_node_id = get_node_id(segment.start.real, segment.start.imag)

            for i in range(1, num_samples + 1):
                t = i / num_samples
                point = segment.point(t)
                current_node_id = get_node_id(point.real, point.imag)

                # Add edge, preventing self-loops
                if prev_node_id != current_node_id:
                    edges.append({
                        'source': prev_node_id,
                        'target': current_node_id
                    })
                prev_node_id = current_node_id

    return raw_nodes, edges


def normalize_and_scale_graph(raw_nodes, max_width, max_length, center_y, center_z):
    """Scales and centers the nodes into the requested bounding box constraint."""
    if not raw_nodes:
        return []

    # 1. Find the current bounding box of the raw geometry
    min_y = min(n['y'] for n in raw_nodes)
    max_y = max(n['y'] for n in raw_nodes)
    min_z = min(n['z'] for n in raw_nodes)
    max_z = max(n['z'] for n in raw_nodes)

    current_width = max_y - min_y
    current_length = max_z - min_z

    # 2. Find the center point of the raw geometry
    raw_center_y = (min_y + max_y) / 2.0
    raw_center_z = (min_z + max_z) / 2.0

    # 3. Calculate uniform scaling factor to fit within max_width and max_length
    scale_y = (max_width / current_width) if current_width > 0 else 1.0
    scale_z = (max_length / current_length) if current_length > 0 else 1.0
    scale = min(scale_y, scale_z)  # Uniform scale prevents stretching/distortion

    # 4. Apply transformations to all nodes
    final_nodes = []
    for i, node in enumerate(raw_nodes):
        # Translate to origin, scale, then translate to target center
        new_y = (node['y'] - raw_center_y) * scale + center_y
        new_z = (node['z'] - raw_center_z) * scale + center_z

        final_nodes.append({
            'id': i,
            'x': 0.0,
            'y': round(new_y, 4),
            'z': -round(new_z, 4)
        })

    return final_nodes


def visualize_graph(nodes, edges):
    """Plots the graph nodes and edges using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nError: 'matplotlib' is required for visualization.")
        print("Please install it by running: pip install matplotlib")
        return

    print("Opening 2D visualizer...")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a dictionary for O(1) node coordinate lookups
    node_dict = {n['id']: n for n in nodes}

    # Plot edges
    for edge in edges:
        src = node_dict.get(edge['source'])
        tgt = node_dict.get(edge['target'])
        if src and tgt:
            ax.plot([src['y'], tgt['y']], [src['z'], tgt['z']], color='gray', linewidth=1, zorder=1)

    # Plot nodes
    y_coords = [n['y'] for n in nodes]
    z_coords = [n['z'] for n in nodes]
    ax.scatter(y_coords, z_coords, c='blue', s=15, zorder=2)

    # Formatting
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Y (Width)')
    ax.set_ylabel('Z (Length)')
    ax.set_title(f"Graph Visualization: {len(nodes)} nodes, {len(edges)} edges")
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Convert an SVG file into a scaled 3D graph (YAML).")
    parser.add_argument("-i", "--input", required=True, help="Path to the input SVG file.")
    parser.add_argument("-o", "--output", default="target_graph.yaml", help="Path for the output YAML file.")
    parser.add_argument("-mw", "--max-width", type=float, default=2.0,
                        help="Maximum width (Y-axis bounds) for the scaled output.")
    parser.add_argument("-ml", "--max-length", type=float, default=1.0,
                        help="Maximum length (Z-axis bounds) for the scaled output.")
    parser.add_argument("-cy", "--center-y", type=float, default=0.0,
                        help="Target Y coordinate for the center of the graph (default: 0.0).")
    parser.add_argument("-cz", "--center-z", type=float, default=0.0,
                        help="Target Z coordinate for the center of the graph (default: 0.0).")
    parser.add_argument("--visualize", action="store_true",
                        help="Open a 2D plot of the generated graph nodes and edges.")

    args = parser.parse_args()

    print(f"Reading SVG from: {args.input}")
    paths = extract_paths_from_svg(args.input)

    if not paths:
        print("No <path> elements found in the SVG.")
        return

    print(f"Found {len(paths)} path elements. Building raw graph...")
    raw_nodes, edges = build_raw_graph(paths)

    print(
        f"Normalizing and scaling (Max Width: {args.max_width}, Max Length: {args.max_length}, Center: {args.center_y}, {args.center_z})...")
    final_nodes = normalize_and_scale_graph(
        raw_nodes,
        args.max_width,
        args.max_length,
        args.center_y,
        args.center_z
    )

    # Prepare data for YAML export
    graph_data = {
        'nodes': final_nodes,
        'edges': edges
    }

    print(f"Writing {len(final_nodes)} nodes and {len(edges)} edges to {args.output}...")
    with open(args.output, "w") as file:
        yaml.dump(graph_data, file, default_flow_style=False, sort_keys=False)

    print("Done!")

    if args.visualize:
        visualize_graph(final_nodes, edges)


if __name__ == "__main__":
    main()
