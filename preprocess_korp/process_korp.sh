#!/bin/bash
# Compute KORP pairwise potential for pdb

if [ $# -ne 2 ]; then
    echo "Usage: ./process_korp.sh <INPUT_ROOT> <OUTPUT_ROOT>"
    exit 1
fi

INPUT_ROOT=$1
OUTPUT_ROOT=$2
KORP_EXECUTABLE="/path/to/your/korpe_gcc"
SCORE_FILE="/path/to/your/korp6Dv1.bin"

# check necessary files and directories
if [[ ! -f "$KORP_EXECUTABLE" ]]; then
    echo "Error: korp_gcc not found: $KORP_EXECUTABLE"
    exit 1
fi
if [ ! -f "$SCORE_FILE" ]; then
    echo "Error: korp6Dv1.bin not found: $SCORE_FILE"
    exit 1
fi
if [ ! -d "$INPUT_ROOT" ]; then
    echo "Error: input root not found: $INPUT_ROOT"
    exit 1
fi

# create output directory
mkdir -p "$OUTPUT_ROOT"

echo "=========================================="
echo "KORP computing start"
echo "=========================================="
echo "Input root: $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Score file: $SCORE_FILE"
echo "=========================================="

# counters
total_files=0
processed_files=0
failed_files=0

# Count all files
for pdb_file in $INPUT_ROOT/*.pdb; do
    if [ -f "$pdb_file" ]; then
        total_files=$((total_files + 1))
    fi
done

echo "Found $total_files PDB files"
echo "=========================================="


# Process all PDB files
for pdb_file in $INPUT_ROOT/*.pdb; do
    if [ -f "$pdb_file" ]; then
        pdb_basename=$(basename "$pdb_file" .pdb)
        processed_files=$((processed_files + 1))
        
        echo "[$processed_files/$total_files] Processing: $pdb_basename"
        
        # Run KORP
        cd "$OUTPUT_ROOT"
        
        if "$KORP_EXECUTABLE" "$pdb_file" \
            --score_file "$SCORE_FILE" \
            -o "$pdb_basename"; then
            echo "  ✓ Success: $pdb_basename"
        else
            echo "  ✗ Failed: $pdb_basename"
            failed_files=$((failed_files + 1))
        fi
        echo "  ------------------------------------------"
    fi
done

echo "=========================================="
echo "KORP computing finished"
echo "=========================================="
echo "Input root: $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Total num of files: $total_files"
echo "Success: $((processed_files - failed_files))"
echo "Failed: $failed_files"
echo "=========================================="