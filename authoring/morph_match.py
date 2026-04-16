import sys
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 morph_match.py <p1_json_path> <p2_json_path> <out_json_path>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        p1 = json.load(f)
    with open(sys.argv[2], 'r') as f:
        p2 = json.load(f)

    # p1 lists drones: list of dicts: {'id': int, 'name': str, 'x': float, 'y': float, 'z': float, 'type': 'H'/'V'}
    # p2 lists targets: list of dicts: {'x', 'y', 'z', 'types': ['H', 'V', ...], 'length_1', 'length_2', 'angle_1', 'angle_2', 'yaw'}

    n1 = len(p1)
    n2 = len(p2)

    if n1 == 0 or n2 == 0:
        with open(sys.argv[3], 'w') as f:
            json.dump([], f)
        sys.exit(0)

    # Cost matrix
    cost = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            d1 = p1[i]
            t2 = p2[j]

            # Check if type is allowed
            if d1['type'] not in t2['types']:
                cost[i, j] = 1e9  # Forbidden
            else:
                dx = d1['x'] - t2['x']
                dy = d1['y'] - t2['y']
                dz = d1['z'] - t2['z']
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = []
    # Only keep assignments that are valid (cost < 1e8)
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < 1e8:
            assn = {
                'drone_id': p1[i]['id'],
                'drone_name': p1[i]['name'],
                'target': p2[j],
                'cost': cost[i, j],
                'swapped': False
            }
            # Determine if angles needed swapping based on assigned type
            # We recreate the angle check logic for the assigned type
            a1 = p2[j]['angle_1']
            a2 = p2[j]['angle_2']
            t = d1['type']
            
            # (Just doing a simple pass-through of the target data, we can figure out exact angles in blender_addon.py based on type matching logic again)
            
            assignments.append(assn)

    with open(sys.argv[3], 'w') as f:
        json.dump(assignments, f)

if __name__ == '__main__':
    main()
