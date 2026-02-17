#!/bin/bash

# Default Input files (if not provided as args)
INPUT_FILES=("authoring/s.yaml")

# If arguments provided, use them as input files
if [ "$#" -gt 0 ]; then
    INPUT_FILES=("$@")
fi

# Directory for results
mkdir -p authoring/results
mkdir -p authoring/results/2d
mkdir -p authoring/results/3d


# --- Configuration Arrays ---
SELECTION_METHODS=("BRUTE_FORCE" "GREEDY_MAX_DEGREE" "GREEDY_TOP_Z" "GREEDY_BOTTOM_Z" "RANDOM")

RESOLUTION_ORDERS=("MAX_DEGREE" "TOP_Z" "BOTTOM_Z" "RANDOM")
# "SAME_AS_PHASE_2" was omitted from the above list

TRAJECTORY_TYPES=("POINT_SPECIFIC" "GLOBAL_CENTROID")

MOVE_DIRECTIONS=("AWAY_FROM_CAMERA" "TOWARDS_CAMERA" "HYBRID")

#PLACEMENT_TYPES=("LAYERS")
PLACEMENT_TYPES=("MIN_DISTANCE" "LAYERS")

# --- Execution ---

for input_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file '$input_file' not found. Skipping."
        continue
    fi

    echo "========================================"
    echo "Processing Input: $input_file"
    echo "========================================"

    for sel in "${SELECTION_METHODS[@]}"; do
        for res in "${RESOLUTION_ORDERS[@]}"; do
            for traj in "${TRAJECTORY_TYPES[@]}"; do
                for move in "${MOVE_DIRECTIONS[@]}"; do
                    for place in "${PLACEMENT_TYPES[@]}"; do

                        # Construct a unique output filename based on parameters
                        # Abbreviate names to keep filename length reasonable
                        # e.g., GREEDY_MAX_DEGREE -> GMD

                        comb_name="$(basename "$input_file" .yaml)_${sel}_${res}_${traj}_${move}_${place}"
                        out_name="authoring/results/out_${comb_name}.yaml"
                        viz_2d_out_name="authoring/results/2d/2d_viz_${comb_name}.png"
                        viz_3d_out_name="authoring/results/3d/3d_viz_${comb_name}.png"

                        echo "Running Config: Sel=$sel Res=$res Traj=$traj Move=$move Place=$place"

                        python authoring/deconflict.py \
                            --input_file "$input_file" \
                            --output_file "$out_name" \
                            --viz_2d_output_file "$viz_2d_out_name" \
                            --viz_3d_output_file "$viz_3d_out_name" \
                            --selection_method "$sel" \
                            --resolution_order "$res" \
                            --trajectory_type "$traj" \
                            --move_direction "$move" \
                            --placement_type "$place" \
                            --save_viz

                    done
                done
            done
        done
    done
done

echo "All combinations completed."