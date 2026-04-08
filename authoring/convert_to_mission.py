import yaml
import sys
import argparse
import os

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Convert deconflict output to mission YAML")
    parser.add_argument('--input', required=True, help="Path to deconflict_output.yaml")
    parser.add_argument('--output', required=True, help="Path to output blender_mission.yaml")
    parser.add_argument('--manifest', required=True, help="Path to swarm_manifest.yaml")
    parser.add_argument('--color', default="[46, 204, 113]", help="RGB color array string like '[46, 204, 113]'")
    parser.add_argument('--mission_name', default="Auto_Generated_Mission", help="Name of the mission")
    
    args = parser.parse_args()

    try:
        points_data = load_yaml(args.input)
    except Exception as e:
        print(f"Error loading input YAML {args.input}: {e}")
        sys.exit(1)

    points = points_data.get('points', [])
    
    # Process manifest
    inventory = {'H': [], 'V': []}
    if os.path.exists(args.manifest):
        try:
            manifest_data = load_yaml(args.manifest)
            for d in manifest_data.get('drones', []):
                t = d.get('type')
                if t in ('H', 'V'):
                    inventory[t].append(d['id'])
        except Exception as e:
            print(f"Error loading manifest {args.manifest}: {e}")
    else:
        print(f"Warning: Manifest not found at {args.manifest}. Using generated IDs.")

    # Assignment logic
    def get_type_matches(a1, a2, l1, l2):
        def fits_H(a, b):
            am, bm = a % 360, b % 360
            ok_a = (l1 == 0) or (0 <= am <= 180)
            bs = 360 if bm == 0 else bm
            ok_b = (l2 == 0) or (180 <= bs <= 360)
            if ok_a and ok_b: return (am, bs)
            return None
            
        def fits_V(a, b):
            am, bm = a % 360, b % 360
            ok_a = (l1 == 0) or (90 <= am <= 270)
            bs = bm if bm > 270 else bm + 360
            ok_b = (l2 == 0) or (270 <= bs <= 450)
            if ok_a and ok_b: return (am, bs)
            return None

        matches = {}
        res_h = fits_H(a1, a2)
        if res_h: matches['H'] = (res_h[0], res_h[1], l1, l2, False)
        res_v = fits_V(a1, a2)
        if res_v: matches['V'] = (res_v[0], res_v[1], l1, l2, False)
        
        res_h_s = fits_H(a2, a1)
        if res_h_s and 'H' not in matches: matches['H'] = (res_h_s[0], res_h_s[1], l2, l1, True)
        res_v_s = fits_V(a2, a1)
        if res_v_s and 'V' not in matches: matches['V'] = (res_v_s[0], res_v_s[1], l2, l1, True)
        
        return matches

    assignments = [None] * len(points)
    ambiguous = []
    created_counts = {'H': 0, 'V': 0}
    
    for i, pt in enumerate(points):
        matches = get_type_matches(pt['angle_1'], pt['angle_2'], pt['length_1'], pt['length_2'])
        if len(matches) == 1:
            t = list(matches.keys())[0]
            assignments[i] = (t, matches[t])
        elif len(matches) == 0:
            t = 'H' if len(inventory['H']) >= len(inventory['V']) else 'V'
            assignments[i] = (t, (pt['angle_1']%360, pt['angle_2']%360, pt['length_1'], pt['length_2'], False))
        else:
            ambiguous.append((i, matches))
            
    for i, matches in ambiguous:
        t = 'H' if len(inventory['H']) >= len(inventory['V']) else 'V'
        assignments[i] = (t, matches[t])
        if len(inventory[t]) > 0:
            inventory[t].pop(0)
        else:
            created_counts[t] += 1

    # Reload inventory for actual assignment loop
    inventory = {'H': [], 'V': []}
    if os.path.exists(args.manifest):
        try:
            manifest_data = load_yaml(args.manifest)
            for d in manifest_data.get('drones', []):
                t = d.get('type')
                if t in ('H', 'V'):
                    inventory[t].append(d['id'])
        except:
            pass

    # Build mission
    output_lines = []
    output_lines.append(f"name: {args.mission_name}")
    output_lines.append("drones:")
    
    for i, pt in enumerate(points):
        t, (a1, a2, l1, l2, swapped) = assignments[i]
        
        if len(inventory[t]) > 0:
            lb_id = inventory[t].pop(0)
        else:
            lb_id = f"lb_extra_{t}_{i}"
            
        x = round(pt.get('x', 0.0), 4)
        y = round(pt.get('y', 0.0), 4)
        z = round(pt.get('z', 0.0), 4)
        yaw = round(pt.get('yaw', 0.0), 4)
        
        target = f"[{x}, {y}, {z}, {yaw}]"
        waypoints = f"[{target}]"
        delta_t = 1.0
        
        s1 = round(a1, 2)
        s2 = round(a2, 2)
        servos = f"[[{s1}, {s2}]]"
        
        max_len = pt.get('max_length_limit', 0.16)
        active_1 = round(25 * (l1 / max_len)) if max_len > 0 else 0
        active_2 = round(25 * (l2 / max_len)) if max_len > 0 else 0
        
        p1_val = 25.0 - active_1
        p2_val = 25.0 + active_2
        
        pointers = f"[[{p1_val}, {p2_val}]]"
        
        # Color formula
        base_color = "[0, 0, 0]"
        c1 = args.color
        c2 = "[0, 0, 0]"
        formula = f"({base_color}) if i < p0 else ({c1}) if i < p1 else ({c2})"
        
        output_lines.append(f"  {lb_id}:")
        output_lines.append(f"    target: {target}")
        output_lines.append(f"    waypoints: {waypoints}")
        output_lines.append(f"    delta_t: {delta_t}")
        output_lines.append(f"    iterations: 1")
        output_lines.append(f"    params:")
        output_lines.append(f"      linear: true")
        output_lines.append(f"      relative: false")
        output_lines.append(f"    servos: {servos}")
        output_lines.append(f"    pointers: {pointers}")
        output_lines.append(f"    led:")
        output_lines.append(f"      mode: \"expression\"")
        output_lines.append(f"      rate: 50")
        output_lines.append(f"      formula: \"{formula}\"")
        
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    try:
        with open(args.output, 'w') as f:
            f.write('\n'.join(output_lines) + '\n')
        # print(f"Mission YAML successfully written to {args.output}")
    except Exception as e:
        print(f"Failed to write mission YAML: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
