#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <place_script.py> [input_file.svg ...]"
    echo "Example: $0 place.py svg/c.svg"
    exit 1
fi

PLACE_SCRIPT=$1
shift

# Default Input files
INPUT_FILES=("dtla.svg")

# If arguments provided, use them as input files
if [ "$#" -gt 0 ]; then
    INPUT_FILES=("$@")
fi

# Base Directory for all results
BASE_RESULTS_DIR="results/skyline_split"
mkdir -p "$BASE_RESULTS_DIR"

# --- Transform Parameters ---
TRANSFORM_MAX_WIDTH=2.0
TRANSFORM_MAX_HEIGHT=1.0

# --- Placement Parameters ---
MAX_LENGTH=0.16
MAX_LENGTHS="0.13 0.16 0.24"
MIN_CHUNK_LEN=0.001
PLACEMENT_POLICIES=("VFG")
# PLACEMENT_POLICIES=("VFG" "SC")

# --- Stagger Parameters ---
SELECTION_METHODS=("GREEDY_MAX_DEGREE" "GREEDY_TOP_Z" "GREEDY_BOTTOM_Z" "RANDOM")
#SELECTION_METHODS=("BRUTE_FORCE" "GREEDY_MAX_DEGREE" "GREEDY_TOP_Z" "GREEDY_BOTTOM_Z" "RANDOM")
RESOLUTION_ORDERS=("MAX_DEGREE" "TOP_Z" "BOTTOM_Z" "RANDOM")
#RESOLUTION_ORDERS=("MAX_DEGREE" "TOP_Z" "BOTTOM_Z" "RANDOM")
# TRAJECTORY_TYPES=("POINT_SPECIFIC")
TRAJECTORY_TYPES=("POINT_SPECIFIC" "GLOBAL_CENTROID")
# MOVE_DIRECTIONS=("HYBRID")
MOVE_DIRECTIONS=("AWAY_FROM_CAMERA" "TOWARDS_CAMERA" "HYBRID")
# DECONFLICT_PLACEMENT_TYPES=("MIN_DISTANCE")
DECONFLICT_PLACEMENT_TYPES=("MIN_DISTANCE" "LAYERS")
ALLOW_SPLIT=true

# Shared Camera Position
CAM_X=2.3
CAM_Y=0.0
CAM_Z=0.0

# Define the comprehensive CSV header
CSV_HEADER="InputFile,TransformNodes,TransformEdges,PlacementPolicy,PlaceExecTime,PlaceTotalLBs,PlaceTotalSegs,PlaceAvgSegLen,PlaceSegLenUtil,SCTotalCand,SCTotalChunks,SCTotalIter,GreedySol,GreedyOverlap,SCOverlap,IsSCBetterThanGreedy,SelectionMethod,ResolutionOrder,TrajectoryType,MoveDirection,DeconflictPlacementType,DownwashConflicts,Collisions,UnresolvedDownwashes,UnresolvedCollisions,InitMinDW,InitMaxDW,InitMinCol,InitMaxCol,InitMinTotal,InitMaxTotal,FinalMinDW,FinalMaxDW,FinalMinCol,FinalMaxCol,FinalMinTotal,FinalMaxTotal,LBsSelected,LBsMoved,AvgDist,MinDist,MaxDist,AddedLbs,NewUtilization,MatchedLines,ImageDiagonal,AvgPosError,AvgWidthError,OverallAvgError,NormalizedError,SimilarityScore"

if [ "$PLACE_SCRIPT" == "place.py" ]; then
    MAX_LENGTH_REPORT="LB Length                : $MAX_LENGTH"
else
    MAX_LENGTH_REPORT="LB Lengths               : $MAX_LENGTHS"
fi

# --- Generate Configuration Report ---
REPORT_FILE="$BASE_RESULTS_DIR/experiment_config.txt"
cat <<EOF > "$REPORT_FILE"
========================================
        EXPERIMENT CONFIG
========================================
Date/Time : $(date)
Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')
Git Hash  : $(git rev-parse HEAD 2>/dev/null || echo 'N/A')
----------------------------------------
Inputs: ${INPUT_FILES[*]}
----------------------------------------
Transform
Max Width                : $TRANSFORM_MAX_WIDTH
Max Height               : $TRANSFORM_MAX_HEIGHT
----------------------------------------
Placement ($PLACE_SCRIPT)
$MAX_LENGTH_REPORT
Placement Policies       : ${PLACEMENT_POLICIES[*]}
SC Min Chunk Length      : $MIN_CHUNK_LEN
----------------------------------------
Stagger
Selection Methods        : ${SELECTION_METHODS[*]}
Resolution Orders        : ${RESOLUTION_ORDERS[*]}
Trajectory Types         : ${TRAJECTORY_TYPES[*]}
Move Directions          : ${MOVE_DIRECTIONS[*]}
Placement Type           : ${DECONFLICT_PLACEMENT_TYPES[*]}
Allow Split              : $ALLOW_SPLIT
----------------------------------------
Perspective Camera
Camera Position X        : $CAM_X
Camera Position Y        : $CAM_Y
Camera Position Z        : $CAM_Z
========================================
EOF
echo "Generated configuration report: $REPORT_FILE"

# --- Execution ---

for input_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file '$input_file' not found. Skipping."
        continue
    fi

    # Extract filename without extension (e.g., "drawing" from "drawing.svg")
    FILE_BASENAME=$(basename "$input_file" .svg)

    echo "========================================"
    echo "Processing Input: $input_file"
    echo "========================================"

    # Setup base directory for this input file
    FILE_DIR="$BASE_RESULTS_DIR/$FILE_BASENAME"
    mkdir -p "$FILE_DIR/yaml"

    # Initialize CSV for this file
    CSV_FILE="$FILE_DIR/${FILE_BASENAME}.csv"
    echo "$CSV_HEADER" > "$CSV_FILE"

    # ---------------------------------------------------------
    # STEP 1: TRANSFORM (SVG -> Graph YAML)
    # ---------------------------------------------------------
    GRAPH_YAML="$FILE_DIR/yaml/graph.yaml"
    echo "  [Step 1] Transforming SVG to Graph..."

    # Capture output to extract metrics (expecting CSV format on the last line)
    TRANSFORM_OUT=$(python transform.py \
        --input "$input_file" \
        --output "$GRAPH_YAML" \
        -mw "$TRANSFORM_MAX_WIDTH" \
        -ml "$TRANSFORM_MAX_HEIGHT" \
        --csv)

    if [ ! -f "$GRAPH_YAML" ]; then
        echo "$TRANSFORM_OUT"
        echo "    Error: transform.py failed to produce $GRAPH_YAML. Skipping to next file."
        continue
    fi

    # Extract the whole CSV line
    TRANSFORM_STATS=$(echo "$TRANSFORM_OUT" | tail -n 1)
    if [[ "$TRANSFORM_STATS" != *","* ]]; then
        TRANSFORM_STATS="0,0"
    fi

    # Loop over Placement Policies
    for policy in "${PLACEMENT_POLICIES[@]}"; do

        # Setup specific directory structure for this policy
        POLICY_DIR="$FILE_DIR/$policy"
        mkdir -p "$POLICY_DIR/2d"
        mkdir -p "$POLICY_DIR/3d"
        mkdir -p "$POLICY_DIR/graph_viz"
        mkdir -p "$POLICY_DIR/bar_viz"
        mkdir -p "$POLICY_DIR/svg"
        mkdir -p "$POLICY_DIR/yaml"

        # ---------------------------------------------------------
        # STEP 2: PLACE (Graph YAML -> Initial Layout YAML)
        # ---------------------------------------------------------
        INITIAL_LAYOUT="$POLICY_DIR/yaml/initial_layout.yaml"
        echo "  [Step 2] Running Placement ($policy) via $PLACE_SCRIPT..."

        if [ "$PLACE_SCRIPT" == "place.py" ]; then
            PLACE_LENGTH_ARGS="--max_len $MAX_LENGTH"
        else
            PLACE_LENGTH_ARGS="--max_lens $MAX_LENGTHS"
        fi

        # Capture output
        PLACE_OUT=$(python $PLACE_SCRIPT \
            --input "$GRAPH_YAML" \
            --output "$INITIAL_LAYOUT" \
            --policy "$policy" \
            $PLACE_LENGTH_ARGS \
            --min_chunck_len $MIN_CHUNK_LEN \
            --no_viz \
            --csv)

        # echo "$PLACE_OUT"
        if [ ! -f "$INITIAL_LAYOUT" ]; then
            echo "$PLACE_OUT"
            echo "    Error: place.py failed to produce $INITIAL_LAYOUT. Skipping policy $policy."
            continue
        fi

        # Use whole lines from place.py depending on the policy
        if [ "$policy" == "SC" ]; then
            PLACE_SC_METRICS=$(echo "$PLACE_OUT" | tail -n 2 | head -n 1)
            PLACE_STD_METRICS=$(echo "$PLACE_OUT" | tail -n 1)
            PLACE_STATS="${PLACE_STD_METRICS},${PLACE_SC_METRICS}"
        else
            PLACE_STD_METRICS=$(echo "$PLACE_OUT" | tail -n 1)
            PLACE_STATS="${PLACE_STD_METRICS},NA,NA,NA,NA,NA,NA,NA"
        fi

        # ---------------------------------------------------------
        # STEP 2.5: REFERENCE SVG RENDER
        # ---------------------------------------------------------
        REF_SVG="$POLICY_DIR/svg/reference_initial.svg"
        echo "  [Step 2.5] Rendering Reference SVG for Comparison..."
        python perspective_camera.py \
            --action render \
            --input "$INITIAL_LAYOUT" \
            --output "$REF_SVG" \
            --camera_pos $CAM_X $CAM_Y $CAM_Z

        if [ "$ALLOW_SPLIT" == "true" ]; then
            SPLIT_ARG="--allow-split"
        else
            SPLIT_ARG=""
        fi

        # Iterate Combinations for Deconflict
        for sel in "${SELECTION_METHODS[@]}"; do
            for res in "${RESOLUTION_ORDERS[@]}"; do
                for traj in "${TRAJECTORY_TYPES[@]}"; do
                    for move in "${MOVE_DIRECTIONS[@]}"; do
                        for d_place in "${DECONFLICT_PLACEMENT_TYPES[@]}"; do

                            # Define configuration ID and paths
                            config_id="${sel}_${res}_${traj}_${move}_${d_place}"

                            out_yaml="$POLICY_DIR/yaml/feasible_${config_id}.yaml"
                            out_svg="$POLICY_DIR/svg/result_${config_id}.svg"
                            out_2d="$POLICY_DIR/2d/${config_id}.png"
                            out_3d="$POLICY_DIR/3d/${config_id}.png"
                            out_graph_viz="$POLICY_DIR/graph_viz/${config_id}.png"
                            out_bar_viz="$POLICY_DIR/bar_viz/${config_id}.png"

                            echo "    [Step 3 & 4] Deconflict & Render: $policy + $config_id"

                            # ---------------------------------------------------------
                            # STEP 3: DECONFLICT (Initial Layout -> Feasible Layout)
                            # ---------------------------------------------------------
                            SOLVER_STATS=$(python deconflict.py \
                                --input_file "$INITIAL_LAYOUT" \
                                --output_file "$out_yaml" \
                                --selection_method "$sel" \
                                --resolution_order "$res" \
                                --trajectory_type "$traj" \
                                --move_direction "$move" \
                                --placement_type "$d_place" \
                                --camera_pos $CAM_X $CAM_Y $CAM_Z \
                                --viz_2d_output_file "$out_2d" \
                                --viz_3d_output_file "$out_3d" \
                                --viz_graph_output_file "$out_graph_viz" \
                                --viz_bar_output_file "$out_bar_viz" \
                                --save_viz \
                                $SPLIT_ARG \
                                --csv | tail -n 1)

                            # Validate Solver Output (Requires 7 comma-separated numbers)
                            if [[ "$SOLVER_STATS" != *","* ]]; then
                                echo "      Error: Solver failed or returned invalid CSV."
                                SOLVER_STATS="0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0" # Dummy data
                            fi

                            # ---------------------------------------------------------
                            # STEP 4: RENDER & COMPARE (Feasible Layout -> SVG -> Diff)
                            # ---------------------------------------------------------
                            # Render Result SVG
                            python perspective_camera.py \
                                --action render \
                                --input "$out_yaml" \
                                --output "$out_svg" \
                                --camera_pos $CAM_X $CAM_Y $CAM_Z

                            # Compare reference SVG with the output SVG
                            CAMERA_STATS=$(python perspective_camera.py \
                                --action compare \
                                --input "$REF_SVG" \
                                --output "$out_svg" \
                                --csv | tail -n 1)

                            # Validate Camera Output
                            if [[ "$CAMERA_STATS" != *","* ]]; then
                                echo "      Error: Camera comparison failed."
                                CAMERA_STATS="0,0.0,0.0,0.0,0.0,0.0,0.0" # Dummy data
                            fi

                            # Append row to CSV
                            echo "$input_file,$TRANSFORM_STATS,$policy,$PLACE_STATS,$sel,$res,$traj,$move,$d_place,$SOLVER_STATS,$CAMERA_STATS" >> "$CSV_FILE"

                        done
                    done
                done
            done
        done
    done

    echo "  Results saved to $CSV_FILE"
done

# ---------------------------------------------------------
# STEP 5: COMBINE INTO MASTER CSV
# ---------------------------------------------------------
if [ "$PLACE_SCRIPT" == "place.py" ]; then
    MASTER_CSV="$BASE_RESULTS_DIR/master_results_${TRANSFORM_MAX_HEIGHT}.csv"
else
    MASTER_CSV="$BASE_RESULTS_DIR/master_results_multi_type_${TRANSFORM_MAX_HEIGHT}.csv"
fi
echo "========================================"
echo "Compiling Master CSV..."

# Check if any CSVs were generated
if find "$BASE_RESULTS_DIR" -mindepth 2 -name "*.csv" -print -quit | grep -q .; then
    # Write the header to the master file
    echo "$CSV_HEADER" > "$MASTER_CSV"

    # Append all CSV rows (skipping the header row) quietly
    find "$BASE_RESULTS_DIR" -mindepth 2 -name "*.csv" | xargs tail -q -n +2 >> "$MASTER_CSV"
    echo "Master CSV successfully generated at: $MASTER_CSV"
else
    echo "Warning: No individual CSV files found to combine."
fi

echo "========================================"
echo "Experiment Completed."
echo "Results hierarchy created in: $BASE_RESULTS_DIR/"
echo "========================================"