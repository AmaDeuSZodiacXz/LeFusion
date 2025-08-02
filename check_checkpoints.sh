#!/bin/bash

# Check available checkpoints in DiffMask model folder
echo "Checking available checkpoints in DiffMask/DiffMask_Model/..."

checkpoint_dir="DiffMask/DiffMask_Model"

if [ -d "$checkpoint_dir" ]; then
    echo "Found checkpoints:"
    ls -la "$checkpoint_dir"/*.pt 2>/dev/null | while read line; do
        if [[ $line =~ model-([0-9]+)\.pt ]]; then
            step_num=${BASH_REMATCH[1]}
            # Convert milestone to actual step (milestone * 1000)
            actual_step=$((step_num * 1000))
            echo "  - model-${step_num}.pt (Step ${actual_step})"
        elif [[ $line =~ diffmask\.pt ]]; then
            echo "  - diffmask.pt (Latest checkpoint - will be auto-detected)"
        else
            echo "  - $line"
        fi
    done
    
    if [ ! -f "$checkpoint_dir"/*.pt ]; then
        echo "No .pt checkpoint files found in $checkpoint_dir"
    fi
else
    echo "Checkpoint directory $checkpoint_dir not found"
fi

echo ""
echo "To resume training:"
echo "1. Use diffmask_resume_latest.sh to automatically load the latest checkpoint"
echo "2. Use diffmask_resume_train.sh and modify the checkpoint_path variable"
echo "3. Modify train_num_steps in the script to set your target step count" 