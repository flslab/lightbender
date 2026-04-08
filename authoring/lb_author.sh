#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_file.svg>"
    exit 1
fi

INPUT_SVG=$1
if [ ! -f "$INPUT_SVG" ]; then
    echo "Error: Input file '$INPUT_SVG' not found."
    exit 1
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
MANIFEST="/path/to/swarm_manifest.yaml"
MISSION_DIR="/path/to/mission"
MISSION_YAML="$MISSION_DIR/lb_author_mission.yaml"
ORCHESTRATOR="/path/to/orchestrator.py"

# --- Derived paths ---
BASE_DIR="results/lb_author"
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
python transform.py \
    --input "$INPUT_SVG" \
    --output "$GRAPH_YAML" \
    -mw "$TRANSFORM_MAX_WIDTH" \
    -ml "$TRANSFORM_MAX_HEIGHT" \
    -cy "$TRANSFORM_CENTER_X" \
    -cz "$TRANSFORM_CENTER_Z" \
    # --csv > /dev/null

echo "----------------------------------------"
echo "  2. Place (Graph -> Tentative Layout)"
echo "     Input : $GRAPH_YAML"
echo "     Config:"
echo "       Policy                     : $POLICY"
echo "       LightBender Segment Length : $MAX_LENGTH"
echo "     Output: $TENTATIVE_LAYOUT"
echo "----------------------------------------"
python place.py \
    --input "$GRAPH_YAML" \
    --output "$TENTATIVE_LAYOUT" \
    --policy "$POLICY" \
    --max_len "$MAX_LENGTH" \
    --min_chunck_len "$MIN_CHUNK_LEN" \
    --no_viz \
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
python deconflict.py \
    --input_file "$TENTATIVE_LAYOUT" \
    --output_file "$STAGGERED_LAYOUT" \
    --selection_method "$SELECTION" \
    --resolution_order "$RESOLUTION" \
    --trajectory_type "$TRAJECTORY" \
    --move_direction "$MOVE" \
    --placement_type "$PLACEMENT" \
    --camera_pos $CAM_X $CAM_Y $CAM_Z \
    --no_viz \
    # --csv > /dev/null

echo "----------------------------------------"
echo "  4. Convert to SFL File"
echo "     Input : $STAGGERED_LAYOUT"
echo "     Config:"
echo "       Color     : '$COLOR'"
echo "       File Name : '$FILE_BASENAME'"
echo "     Output: $MISSION_YAML"
echo "----------------------------------------"
python convert_to_mission.py \
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

echo "----------------------------------------"
echo "  5. Illuminating Using Orchestrator"
echo "----------------------------------------"
cd "$(dirname "$ORCHESTRATOR")"
python3 orchestrator.py --illumination --skip-confirm
