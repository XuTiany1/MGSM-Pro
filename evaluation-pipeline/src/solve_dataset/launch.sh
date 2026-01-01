#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 launch.yaml"
    exit 1
fi

LAUNCH_YAML=$1

# Extract all config paths (removes comments and empty lines)
CONFIG_PATHS=$(grep -oP '^\s*-\s*\K.*' "$LAUNCH_YAML" | sed '/^$/d')

for CONFIG in $CONFIG_PATHS; do
    echo "Launching job for $CONFIG"
    sbatch run_gemini.sh "$CONFIG"
done
