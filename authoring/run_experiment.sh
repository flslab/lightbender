#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <place_script.py> <base_results_dir> [input_file.svg ...]"
    echo "Example: $0 place.py results/gurobi/a-z svg/c.svg"
    exit 1
fi

PLACE_SCRIPT=$1
BASE_RESULTS_DIR=$2
shift 2
PYTHON_BIN=${PYTHON_BIN:-python3}

# Default Input files
INPUT_FILES=("dtla.svg")

# If arguments provided, use them as input files
if [ "$#" -gt 0 ]; then
    INPUT_FILES=("$@")
fi

# Base Directory for all results
mkdir -p -- "$BASE_RESULTS_DIR"

# --- Transform Parameters ---
# Each entry is "max_width max_height".
TRANSFORM_MAX_SIZE_PAIRS=(
    "2.0 1.0"
    "1.0 0.5"
    "0.5 0.25"
)

# --- Placement Parameters ---
MAX_LENGTH=0.16
MAX_LENGTHS="0.13 0.16 0.24"
MIN_CHUNK_LEN=0.01
PLACEMENT_POLICIES=("SC" "VFG")
# PLACEMENT_POLICIES=("VFG" "SC" "HYB")
SET_COVER_SOLVERS=("bnb" "gurobi") # used only by SC/HYB
GUROBI_MIP_GAP=0.0

# --- Stagger Parameters ---
SELECTION_METHODS=("GREEDY_MAX_DEGREE")
#SELECTION_METHODS=("BRUTE_FORCE" "GREEDY_MAX_DEGREE" "GREEDY_TOP_Z" "GREEDY_BOTTOM_Z" "RANDOM")
RESOLUTION_ORDERS=("MAX_DEGREE")
# RESOLUTION_ORDERS=("MAX_DEGREE" "TOP_Z" "BOTTOM_Z" "RANDOM")
TRAJECTORY_TYPES=("LINE_OF_SIGHT")
# TRAJECTORY_TYPES=("LINE_OF_SIGHT" "GLOBAL_CENTROID")
MOVE_DIRECTIONS=("HYBRID")
# MOVE_DIRECTIONS=("AWAY_FROM_CAMERA" "TOWARDS_CAMERA" "HYBRID")
# DECONFLICT_PLACEMENT_TYPES=("LAYERS")
DECONFLICT_PLACEMENT_TYPES=("MIN_DISTANCE")
ALLOW_SPLIT=false

# Shared Camera Position
CAM_X=3.0
CAM_Y=0.0
CAM_Z=0.0

CAM_d=(0.0 45.0 90.0)
AESTHETIC_RENDER=false
RENDER_SCALE=4.0

CAM_render=()
for cam_d in "${CAM_d[@]}"; do
    CAM_render+=("$("$PYTHON_BIN" -c 'import math, sys; x, y, z, d = map(float, sys.argv[1:]); a = math.radians(d); print(*(0.0 if abs(v) < 1e-12 else round(v, 10) for v in (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), z)))' "$CAM_X" "$CAM_Y" "$CAM_Z" "$cam_d")")
done

RENDER_STYLE_ARGS=()
if [ "$AESTHETIC_RENDER" == "true" ]; then
    RENDER_STYLE_ARGS=(--aesthetic)
fi

# Define the comprehensive CSV header
CSV_HEADER="InputFile,TransformNodes,TransformEdges,PlacementPolicy,PlaceExecTime,PlaceTotalLBs,PlaceTotalSegs,PlaceAvgSegLen,PlaceSegLenUtil,SetCoverSolver,SCStatus,SCTotalCand,SCTotalChunks,SCTotalNodesOrIter,GreedySol,GreedyOverlap,SCOverlap,IsSCBetterThanGreedy,SelectionMethod,ResolutionOrder,TrajectoryType,MoveDirection,DeconflictPlacementType,DownwashConflicts,Collisions,UnresolvedDownwashes,UnresolvedCollisions,InitMinDW,InitMaxDW,InitMinCol,InitMaxCol,InitMinTotal,InitMaxTotal,FinalMinDW,FinalMaxDW,FinalMinCol,FinalMaxCol,FinalMinTotal,FinalMaxTotal,LBsSelected,LBsMoved,AvgDist,MinDist,MaxDist,AddedLbs,NewUtilization,CameraRotationDeg,RenderCameraX,RenderCameraY,RenderCameraZ,MatchedLines,ImageDiagonal,AvgPosError,AvgWidthAbsError,AvgWidthRelError,OverallAvgError,NormalizedError,SimilarityScore"

if [ "$PLACE_SCRIPT" == "place.py" ]; then
    MAX_LENGTH_REPORT="LB Length                : $MAX_LENGTH"
else
    MAX_LENGTH_REPORT="LB Lengths               : $MAX_LENGTHS"
fi

printf -v TRANSFORM_SIZE_PAIRS_REPORT '  - %s\n' "${TRANSFORM_MAX_SIZE_PAIRS[@]}"

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
Max Width/Height Pairs:
$TRANSFORM_SIZE_PAIRS_REPORT
----------------------------------------
Placement ($PLACE_SCRIPT)
$MAX_LENGTH_REPORT
Placement Policies       : ${PLACEMENT_POLICIES[*]}
SC Min Chunk Length      : $MIN_CHUNK_LEN
Set Cover Solvers        : ${SET_COVER_SOLVERS[*]}
Gurobi MIP Gap           : $GUROBI_MIP_GAP
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
Render Rotations (deg)   : ${CAM_d[*]}
Render Camera Positions  : ${CAM_render[*]}
Aesthetic Render         : $AESTHETIC_RENDER
Render Scale             : $RENDER_SCALE
========================================
EOF
echo "Generated configuration report: $REPORT_FILE"

# --- Execution ---

for transform_size_pair in "${TRANSFORM_MAX_SIZE_PAIRS[@]}"; do
    read -r TRANSFORM_MAX_WIDTH TRANSFORM_MAX_HEIGHT extra_dimension <<< "$transform_size_pair"
    if [ -z "$TRANSFORM_MAX_WIDTH" ] || [ -z "$TRANSFORM_MAX_HEIGHT" ] || [ -n "$extra_dimension" ]; then
        echo "Error: Invalid transform size pair '$transform_size_pair'. Expected: \"max_width max_height\"."
        exit 1
    fi

    TRANSFORM_ID="width_${TRANSFORM_MAX_WIDTH}_height_${TRANSFORM_MAX_HEIGHT}"
    TRANSFORM_RESULTS_DIR="$BASE_RESULTS_DIR/$TRANSFORM_ID"
    mkdir -p "$TRANSFORM_RESULTS_DIR"

    echo "========================================"
    echo "Transform size: width=$TRANSFORM_MAX_WIDTH, height=$TRANSFORM_MAX_HEIGHT"
    echo "========================================"

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
    FILE_DIR="$TRANSFORM_RESULTS_DIR/$FILE_BASENAME"
    mkdir -p "$FILE_DIR/yaml"

    # Initialize CSV for this file
    CSV_FILE="$FILE_DIR/${FILE_BASENAME}_${TRANSFORM_ID}.csv"
    echo "$CSV_HEADER" > "$CSV_FILE"

    # ---------------------------------------------------------
    # STEP 1: TRANSFORM (SVG -> Graph YAML)
    # ---------------------------------------------------------
    GRAPH_YAML="$FILE_DIR/yaml/graph.yaml"
    echo "  [Step 1] Transforming SVG to Graph..."

    # Capture output to extract metrics (expecting CSV format on the last line)
    TRANSFORM_OUT=$("$PYTHON_BIN" transform.py \
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
        if [ "$policy" == "SC" ] || [ "$policy" == "HYB" ]; then
            policy_solvers=("${SET_COVER_SOLVERS[@]}")
        else
            # The solver is unused by non-set-cover policies, so run them only once.
            policy_solvers=("${SET_COVER_SOLVERS[0]}")
        fi

        for set_cover_solver in "${policy_solvers[@]}"; do

        policy_output_name="$policy"
        if [ "$policy" == "SC" ] || [ "$policy" == "HYB" ]; then
            policy_output_name="${policy}-${set_cover_solver}"
        fi

        # Setup specific directory structure for this policy
        POLICY_DIR="$FILE_DIR/$policy_output_name"
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
        SC_LOG="$POLICY_DIR/yaml/sc_log.json"
        echo "  [Step 2] Running Placement ($policy_output_name) via $PLACE_SCRIPT..."

        if [ "$PLACE_SCRIPT" == "place.py" ]; then
            PLACE_LENGTH_ARGS="--max_len $MAX_LENGTH"
        else
            PLACE_LENGTH_ARGS="--max_lens $MAX_LENGTHS"
        fi

        # Capture output
        if ! PLACE_OUT=$("$PYTHON_BIN" "$PLACE_SCRIPT" \
            --input "$GRAPH_YAML" \
            --output "$INITIAL_LAYOUT" \
            --policy "$policy" \
            $PLACE_LENGTH_ARGS \
            --min_chunck_len $MIN_CHUNK_LEN \
            --no_viz \
            --set_cover_log "$SC_LOG" \
            --set_cover_solver "$set_cover_solver" \
            --gurobi_mip_gap "$GUROBI_MIP_GAP" \
            --csv); then
            echo "$PLACE_OUT"
            echo "    Error: placement command failed for $policy_output_name. Skipping."
            continue
        fi

        if [ ! -f "$INITIAL_LAYOUT" ]; then
            echo "$PLACE_OUT"
            echo "    Error: place.py failed to produce $INITIAL_LAYOUT. Skipping policy $policy."
            continue
        fi

        # Use whole lines from place.py depending on the policy
        if [ "$policy" == "SC" ] || [ "$policy" == "HYB" ]; then
            PLACE_SC_METRICS=$(echo "$PLACE_OUT" | tail -n 2 | head -n 1)
            PLACE_STD_METRICS=$(echo "$PLACE_OUT" | tail -n 1)
            PLACE_STATS="${PLACE_STD_METRICS},${PLACE_SC_METRICS}"
        else
            PLACE_STD_METRICS=$(echo "$PLACE_OUT" | tail -n 1)
            PLACE_STATS="${PLACE_STD_METRICS},NA,NA,NA,NA,NA,NA,NA,NA,NA"
        fi

        # ---------------------------------------------------------
        # STEP 2.5: REFERENCE SVG RENDER
        # ---------------------------------------------------------
        echo "  [Step 2.5] Rendering Reference SVGs for Comparison..."
        for cam_idx in "${!CAM_d[@]}"; do
            cam_d="${CAM_d[$cam_idx]}"
            camera_pos="${CAM_render[$cam_idx]}"
            read -r CAM_X_render CAM_Y_render CAM_Z_render <<< "$camera_pos"
            ref_svg="$POLICY_DIR/svg/reference_initial_camera_${cam_d}deg.svg"

            "$PYTHON_BIN" perspective_camera.py \
                --action render \
                --input "$INITIAL_LAYOUT" \
                --output "$ref_svg" \
                --camera_pos "$CAM_X_render" "$CAM_Y_render" "$CAM_Z_render" \
                --render_scale "$RENDER_SCALE" \
                "${RENDER_STYLE_ARGS[@]}"
        done

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
                            out_2d="$POLICY_DIR/2d/${config_id}.png"
                            out_3d="$POLICY_DIR/3d/${config_id}.png"
                            out_graph_viz="$POLICY_DIR/graph_viz/${config_id}.png"
                            out_bar_viz="$POLICY_DIR/bar_viz/${config_id}.png"

                            echo "    [Step 3 & 4] Deconflict & Render: $policy_output_name + $config_id"

                            # ---------------------------------------------------------
                            # STEP 3: DECONFLICT (Initial Layout -> Feasible Layout)
                            # ---------------------------------------------------------
                            SOLVER_STATS=$("$PYTHON_BIN" deconflict.py \
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
                            echo "SOLVER_STATS: $SOLVER_STATS"
                            if [[ "$SOLVER_STATS" != *","* ]]; then
                                echo "      Error: Solver failed or returned invalid CSV."
                                SOLVER_STATS="0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0" # Dummy data
                            fi

                            # ---------------------------------------------------------
                            # STEP 4: RENDER & COMPARE (Feasible Layout -> SVG -> Diff)
                            # ---------------------------------------------------------
                            for cam_idx in "${!CAM_d[@]}"; do
                                cam_d="${CAM_d[$cam_idx]}"
                                camera_pos="${CAM_render[$cam_idx]}"
                                read -r CAM_X_render CAM_Y_render CAM_Z_render <<< "$camera_pos"
                                ref_svg="$POLICY_DIR/svg/reference_initial_camera_${cam_d}deg.svg"
                                out_svg="$POLICY_DIR/svg/result_${config_id}_camera_${cam_d}deg.svg"

                                # Render Result SVG
                                "$PYTHON_BIN" perspective_camera.py \
                                    --action render \
                                    --input "$out_yaml" \
                                    --output "$out_svg" \
                                    --camera_pos "$CAM_X_render" "$CAM_Y_render" "$CAM_Z_render" \
                                    --render_scale "$RENDER_SCALE" \
                                    "${RENDER_STYLE_ARGS[@]}"

                                # Compare reference SVG with the output SVG
                                CAMERA_STATS=$("$PYTHON_BIN" perspective_camera.py \
                                    --action compare \
                                    --input "$ref_svg" \
                                    --output "$out_svg" \
                                    --csv | tail -n 1)

                                # Validate Camera Output
                                if [[ "$CAMERA_STATS" != *","* ]]; then
                                    echo "      Error: Camera comparison failed."
                                    CAMERA_STATS="0,0.0,0.0,0.0,0.0,0.0" # Dummy data
                                fi

                                # Append one row per camera rotation
                                echo "$input_file,$TRANSFORM_STATS,$policy_output_name,$PLACE_STATS,$sel,$res,$traj,$move,$d_place,$SOLVER_STATS,$cam_d,${camera_pos// /,},$CAMERA_STATS" >> "$CSV_FILE"
                            done

                        done
                    done
                done
            done
        done
        done
    done

        echo "  Results saved to $CSV_FILE"
    done

    # ---------------------------------------------------------
    # STEP 5: COMBINE THIS TRANSFORM PAIR INTO A MASTER CSV
    # ---------------------------------------------------------
    if [ "$PLACE_SCRIPT" == "place.py" ]; then
        MASTER_CSV="$BASE_RESULTS_DIR/master_results_${TRANSFORM_ID}.csv"
    else
        MASTER_CSV="$BASE_RESULTS_DIR/master_results_multi_type_${TRANSFORM_ID}.csv"
    fi
    echo "========================================"
    echo "Compiling master CSV for width=$TRANSFORM_MAX_WIDTH, height=$TRANSFORM_MAX_HEIGHT..."

    # Check if any per-input CSVs were generated for this transform pair.
    if find "$TRANSFORM_RESULTS_DIR" -mindepth 2 -name "*.csv" -print -quit | grep -q .; then
        # Write the header to the master file.
        echo "$CSV_HEADER" > "$MASTER_CSV"

        # Append all CSV rows for this pair, skipping their header rows.
        find "$TRANSFORM_RESULTS_DIR" -mindepth 2 -name "*.csv" -exec tail -q -n +2 {} + >> "$MASTER_CSV"
        echo "Master CSV successfully generated at: $MASTER_CSV"
    else
        echo "Warning: No individual CSV files found for $TRANSFORM_ID."
    fi
done

echo "========================================"
echo "Experiment Completed."
echo "Results hierarchy created in: $BASE_RESULTS_DIR/"
echo "========================================"
