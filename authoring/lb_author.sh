#!/bin/bash
set -e

# Repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Repository root: $REPO_ROOT"

# Default values
NO_VIZ=false
ILLUMINATE=false
INPUT_SVG=""

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-viz) NO_VIZ=true; shift ;;
        --illuminate) ILLUMINATE=true; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) 
            if [ -z "$INPUT_SVG" ]; then
                INPUT_SVG="$1"
            else
                echo "Error: Multiple input files specified: $INPUT_SVG and $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$INPUT_SVG" ]; then
    echo "Usage: $0 [--no-viz] [--illuminate] <input_file.svg>"
    exit 1
fi

if [ ! -f "$INPUT_SVG" ]; then
    echo "Error: Input file '$INPUT_SVG' not found."
    exit 1
fi

TRANSFORM_VIZ="--visualize"
PLACE_VIZ=""
DECONFLICT_VIZ=""

if [ "$NO_VIZ" = true ]; then
    TRANSFORM_VIZ=""
    PLACE_VIZ="--no_viz"
    DECONFLICT_VIZ="--no_viz"
fi

# --- Configuration ---
TRANSFORM_MAX_WIDTH=2.0
TRANSFORM_MAX_HEIGHT=0.14
TRANSFORM_CENTER_X=0.0
TRANSFORM_CENTER_Z=1.0

MAX_LENGTH=0.16
MIN_CHUNK_LEN=0.001
POLICY="VFG"

SELECTION="GREEDY_MAX_DEGREE"
RESOLUTION="MAX_DEGREE"
TRAJECTORY="LINE_OF_SIGHT"
MOVE="HYBRID"
PLACEMENT="MIN_DISTANCE"

CAM_X=2.25
CAM_Y=0.0
CAM_Z=1.0

COLOR="[255, 255, 255]"
MANIFEST="$REPO_ROOT/orchestrator/swarm_manifest.yaml"
MISSION_DIR="$REPO_ROOT/orchestrator/SFL"
MISSION_YAML="$MISSION_DIR/lb_author_mission.yaml"
ORCHESTRATOR="$REPO_ROOT/orchestrator/orchestrator.py"

# --- Derived paths ---
BASE_DIR="$REPO_ROOT/authoring/results/lb_author"
FILE_BASENAME=$(basename "$INPUT_SVG" .svg)
OUT_DIR="$BASE_DIR/$FILE_BASENAME"
mkdir -p "$OUT_DIR"

GRAPH_YAML="$OUT_DIR/graph.yaml"
TENTATIVE_LAYOUT="$OUT_DIR/tentative_layout.yaml"
STAGGERED_LAYOUT="$OUT_DIR/staggered_layout.yaml"


echo "========================================"
echo "    Executing End-to-End Pipeline"
echo "========================================"

echo "----------------------------------------"
echo "  1. Transform (SVG -> Graph)"
echo "     Input : $INPUT_SVG"
echo "     Config:"
echo "       Max Width      : $TRANSFORM_MAX_WIDTH"
echo "       Max Height     : $TRANSFORM_MAX_HEIGHT"
echo "       Center         : ($TRANSFORM_CENTER_X, $TRANSFORM_CENTER_Z)"
echo "     Output: $GRAPH_YAML"
echo "----------------------------------------"
if [ ! -f "$INPUT_SVG" ]; then
    echo "Error: Input file '$INPUT_SVG' not found. Cannot proceed with Step 1 (Transform)."
    exit 1
fi
python $REPO_ROOT/authoring/transform.py \
    --input "$INPUT_SVG" \
    --output "$GRAPH_YAML" \
    -mw "$TRANSFORM_MAX_WIDTH" \
    -ml "$TRANSFORM_MAX_HEIGHT" \
    -cy "$TRANSFORM_CENTER_X" \
    -cz "$TRANSFORM_CENTER_Z" \
    $TRANSFORM_VIZ
    # --csv > /dev/null

echo "----------------------------------------"
echo "  2. Place (Graph -> Tentative Layout)"
echo "     Input : $GRAPH_YAML"
echo "     Config:"
echo "       Policy                     : $POLICY"
echo "       LightBender Segment Length : $MAX_LENGTH"
echo "     Output: $TENTATIVE_LAYOUT"
echo "----------------------------------------"
if [ ! -f "$GRAPH_YAML" ]; then
    echo "Error: Input file '$GRAPH_YAML' not found. Cannot proceed with Step 2 (Place)."
    exit 1
fi
python $REPO_ROOT/authoring/place.py \
    --input "$GRAPH_YAML" \
    --output "$TENTATIVE_LAYOUT" \
    --policy "$POLICY" \
    --max_len "$MAX_LENGTH" \
    --min_chunck_len "$MIN_CHUNK_LEN" \
    $PLACE_VIZ \
    # --csv > /dev/null

echo "----------------------------------------"
echo "  3. Stagger (Tentative Layout -> Staggered Layout)"
echo "     Input : $TENTATIVE_LAYOUT"
echo "     Config:"
echo "       Selection      : $SELECTION"
echo "       Ordering       : $RESOLUTION"
echo "       Trajectory     : $TRAJECTORY"
echo "       Direction      : $MOVE"
echo "       Distance       : $PLACEMENT"
echo "       Viewpoint      : ($CAM_X, $CAM_Y, $CAM_Z)"
echo "     Output: $STAGGERED_LAYOUT"
echo "----------------------------------------"
if [ ! -f "$TENTATIVE_LAYOUT" ]; then
    echo "Error: Input file '$TENTATIVE_LAYOUT' not found. Cannot proceed with Step 3 (Stagger)."
    exit 1
fi
python $REPO_ROOT/authoring/deconflict.py \
    --input_file "$TENTATIVE_LAYOUT" \
    --output_file "$STAGGERED_LAYOUT" \
    --selection_method "$SELECTION" \
    --resolution_order "$RESOLUTION" \
    --trajectory_type "$TRAJECTORY" \
    --move_direction "$MOVE" \
    --placement_type "$PLACEMENT" \
    --camera_pos $CAM_X $CAM_Y $CAM_Z \
    $DECONFLICT_VIZ \
    # --csv > /dev/null

echo "----------------------------------------"
echo "  4. Convert to SFL File"
echo "     Input : $STAGGERED_LAYOUT"
echo "     Config:"
echo "       Color     : '$COLOR'"
echo "       File Name : '$FILE_BASENAME'"
echo "     Output: $MISSION_YAML"
echo "----------------------------------------"
if [ ! -f "$STAGGERED_LAYOUT" ]; then
    echo "Error: Input file '$STAGGERED_LAYOUT' not found. Cannot proceed with Step 4 (Convert)."
    exit 1
fi
python $REPO_ROOT/authoring/convert_to_mission.py \
    --input "$STAGGERED_LAYOUT" \
    --output "$MISSION_YAML" \
    --manifest "$MANIFEST" \
    --color "$COLOR" \
    --mission_name "Experiment_$FILE_BASENAME"

if [ ! -f "$MISSION_YAML" ]; then
    echo "Error: Mission YAML was not generated at $MISSION_YAML"
    exit 1
fi

# echo "Mission generated at $MISSION_YAML"

if [ "$ILLUMINATE" = true ]; then
    echo "----------------------------------------"
    echo "  5. Illuminating Using Orchestrator"
    echo "----------------------------------------"
    cd "$(dirname "$ORCHESTRATOR")"
    python $REPO_ROOT/orchestrator/orchestrator.py --illumination --skip-confirm
fi
