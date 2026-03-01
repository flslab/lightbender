#!/bin/bash

# Default Input files
INPUT_FILES=("points_input.yaml")

# If arguments provided, use them as input files
if [ "$#" -gt 0 ]; then
    INPUT_FILES=("$@")
fi

# Base Directory for all results
BASE_RESULTS_DIR="results_structured"
mkdir -p "$BASE_RESULTS_DIR"

# --- Configuration Arrays ---
SELECTION_METHODS=("GREEDY_MAX_DEGREE")
#SELECTION_METHODS=("BRUTE_FORCE" "GREEDY_MAX_DEGREE" "GREEDY_TOP_Z" "GREEDY_BOTTOM_Z" "RANDOM")
RESOLUTION_ORDERS=("MAX_DEGREE")
#RESOLUTION_ORDERS=("MAX_DEGREE" "TOP_Z" "BOTTOM_Z" "RANDOM")
TRAJECTORY_TYPES=("POINT_SPECIFIC")
#TRAJECTORY_TYPES=("POINT_SPECIFIC" "GLOBAL_CENTROID")
MOVE_DIRECTIONS=("HYBRID")
#MOVE_DIRECTIONS=("AWAY_FROM_CAMERA" "TOWARDS_CAMERA" "HYBRID")
PLACEMENT_TYPES=("MIN_DISTANCE")
#PLACEMENT_TYPES=("MIN_DISTANCE" "LAYERS")

# Shared Camera Position (must match defaults in python scripts or be passed explicitly)
CAM_X=2.3
CAM_Y=1.5
CAM_Z=0.8

# --- Execution ---

for input_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file '$input_file' not found. Skipping."
        continue
    fi

    # Extract filename without extension (e.g., "points_input" from "points_input.yaml")
    FILE_BASENAME=$(basename "$input_file" .yaml)

    echo "========================================"
    echo "Processing Input: $input_file"
    echo "========================================"

    # 1. Setup Directory Structure per Input File
    FILE_DIR="$BASE_RESULTS_DIR/$FILE_BASENAME"
    mkdir -p "$FILE_DIR/2d"
    mkdir -p "$FILE_DIR/3d"
    mkdir -p "$FILE_DIR/svg"
    mkdir -p "$FILE_DIR/yaml"

    # 2. Initialize CSV for this file
    CSV_FILE="$FILE_DIR/${FILE_BASENAME}.csv"
    # Write Header
    echo "InputFile,SelectionMethod,ResolutionOrder,TrajectoryType,MoveDirection,PlacementType,DownwahConflicts,Collisions,LBsSelected,LBsMoved,AvgDist,MinDist,MaxDist,MatchedLines,ImageDiagonal,AvgPosError,AvgWidthError,OverallAvgError,NormalizedError,SimilarityScore" > "$CSV_FILE"

    # 3. Generate Reference SVG
    REF_SVG_PATH="$FILE_DIR/svg/ref_${FILE_BASENAME}.svg"

    echo "  Generating Reference SVG: $REF_SVG_PATH"
    python perspective_camera.py \
        --action render \
        --input "$input_file" \
        --output "$REF_SVG_PATH" \
        --camera_pos $CAM_X $CAM_Y $CAM_Z

    # 4. Iterate Combinations
    for sel in "${SELECTION_METHODS[@]}"; do
        for res in "${RESOLUTION_ORDERS[@]}"; do
            for traj in "${TRAJECTORY_TYPES[@]}"; do
                for move in "${MOVE_DIRECTIONS[@]}"; do
                    for place in "${PLACEMENT_TYPES[@]}"; do

                        # Define configuration ID and paths
                        config_id="${sel}_${res}_${traj}_${move}_${place}"

                        out_yaml="$FILE_DIR/yaml/${config_id}.yaml"
                        out_svg="$FILE_DIR/svg/${config_id}.svg"
                        out_2d="$FILE_DIR/2d/${config_id}.png"
                        out_3d="$FILE_DIR/3d/${config_id}.png"

                        echo "  [Running] $config_id"

                        # A. Run Solver & Capture Metrics
                        # Note: Changed --no_viz to --save_viz and added specific output paths
                        SOLVER_STATS=$(python deconflict.py \
                            --input_file "$input_file" \
                            --output_file "$out_yaml" \
                            --selection_method "$sel" \
                            --resolution_order "$res" \
                            --trajectory_type "$traj" \
                            --move_direction "$move" \
                            --placement_type "$place" \
                            --camera_pos $CAM_X $CAM_Y $CAM_Z \
                            --viz_2d_output_file "$out_2d" \
                            --viz_3d_output_file "$out_3d" \
                            --save_viz \
                            --csv | tail -n 1)

                        # Validate Solver Output
                        if [[ "$SOLVER_STATS" != *","* ]]; then
                            echo "    Error: Solver failed or returned invalid CSV."
                            SOLVER_STATS="0,0,0,0,0" # Dummy data
                        fi

                        # B. Render Result SVG
                        python perspective_camera.py \
                            --action render \
                            --input "$out_yaml" \
                            --output "$out_svg" \
                            --camera_pos $CAM_X $CAM_Y $CAM_Z

                        # C. Compare SVGs & Capture Metrics
                        CAMERA_STATS=$(python perspective_camera.py \
                            --action compare \
                            --input "$REF_SVG_PATH" \
                            --output "$out_svg" \
                            --csv | tail -n 1)

                        # Validate Camera Output
                        if [[ "$CAMERA_STATS" != *","* ]]; then
                            echo "    Error: Camera comparison failed."
                            CAMERA_STATS="0,0,0,0,0,0,0" # Dummy data
                        fi

                        # D. Append to File-Specific CSV
                        echo "$input_file,$sel,$res,$traj,$move,$place,$SOLVER_STATS,$CAMERA_STATS" >> "$CSV_FILE"

                    done
                done
            done
        done
    done

    echo "  Results saved to $CSV_FILE"
done

echo "========================================"
echo "Experiment Completed."
echo "Results hierarchy created in: $BASE_RESULTS_DIR/"
echo "========================================"