#!/bin/bash

# Usage: ./run_camera_metrics.sh <BEFORE_YAML> <AFTER_DIR> <OUTPUT_CSV>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <BEFORE_YAML> <AFTER_DIR> <OUTPUT_CSV>"
    exit 1
fi

BEFORE_YAML=$1
AFTER_DIR=$2
OUTPUT_CSV=$3

# Check inputs
if [ ! -f "$BEFORE_YAML" ]; then
    echo "Error: Before file '$BEFORE_YAML' not found."
    exit 1
fi

if [ ! -d "$AFTER_DIR" ]; then
    echo "Error: Directory '$AFTER_DIR' not found."
    exit 1
fi

# Define temporary SVG filename for the reference
TEMP_SVG_BEFORE="${BEFORE_YAML%.yaml}.svg"

echo "Rendering Reference: $BEFORE_YAML -> $TEMP_SVG_BEFORE"
python authoring/perspective_camera.py --action render --input "$BEFORE_YAML" --output "$TEMP_SVG_BEFORE"

# Initialize CSV File with new columns
echo "Filename,MatchedLines,ImageDiagonal,AvgPosError,AvgWidthError,OverallAvgError,NormalizedError,SimilarityScore" > "$OUTPUT_CSV"

echo "Processing results from $AFTER_DIR..."

# Iterate over YAML files in the directory
for yaml_file in "$AFTER_DIR"/*.yaml; do
    if [ ! -f "$yaml_file" ]; then
        continue
    fi

    # Construct unique SVG filename based on the YAML filename
    # e.g., results/config_1.yaml -> results/config_1.svg
    svg_filename="${yaml_file%.yaml}.svg"

    # Render current 'after' file to the unique SVG path
    python authoring/perspective_camera.py --action render --input "$yaml_file" --output "$svg_filename"

    # Compare with reference and get CSV string
    # We capture stdout. stderr handles errors/warnings.
    STATS=$(python authoring/perspective_camera.py --action compare --input "$TEMP_SVG_BEFORE" --output "$svg_filename" --csv)

    # Check if we got valid stats back (non-empty)
    if [ ! -z "$STATS" ]; then
        filename=$(basename "$yaml_file")
        echo "$filename,$STATS" >> "$OUTPUT_CSV"
        # Score is now the 7th element (AvgPos is 3rd, AvgWidth is 4th)
        score=$(echo $STATS | cut -d',' -f7)
        echo "  Processed: $filename -> Score: $score"
    else
        echo "  Failed to compare: $filename"
    fi

done

# Cleanup (Only remove the temporary reference SVG, keep the 'after' SVGs)
#rm "$TEMP_SVG_BEFORE"

echo "========================================"
echo "Done. Metrics saved to $OUTPUT_CSV"
echo "========================================"