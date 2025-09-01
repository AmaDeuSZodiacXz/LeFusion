#!/bin/bash

# NeuralSynth: Normal-to-Pathological Synthesis Pipeline
# Following LeFusion's approach with enhancements

echo "================================"
echo "NeuralSynth Synthesis Pipeline"
echo "================================"

# Set paths
NEURALSYNTH_DIR="/Users/skb/Documents/LeFusion/NeuralSynth"
DATA_DIR="/Users/skb/Documents/LeFusion/data"

# Activate environment if needed
# source ~/miniconda3/bin/activate neuralsynth

# Function to run synthesis for a dataset
run_synthesis() {
    local DATASET=$1
    local NUM_SYNTHETIC=$2
    
    echo ""
    echo "Running synthesis for $DATASET dataset..."
    echo "Generating $NUM_SYNTHETIC synthetic cases per normal case"
    
    python $NEURALSYNTH_DIR/pipeline/normal_to_pathological.py \
        --dataset $DATASET \
        --normal_dir $DATA_DIR/$DATASET/normal \
        --output_dir $DATA_DIR/$DATASET/synthetic_neuralsynth \
        --num_synthetic $NUM_SYNTHETIC \
        --checkpoint $NEURALSYNTH_DIR/checkpoints/neuralsynth_${DATASET}/best_model.pt
    
    if [ $? -eq 0 ]; then
        echo "✓ $DATASET synthesis completed successfully"
    else
        echo "✗ $DATASET synthesis failed"
        return 1
    fi
}

# Main execution
main() {
    # Check if specific dataset is provided
    if [ "$1" = "lidc" ] || [ "$1" = "LIDC" ]; then
        echo "Processing LIDC dataset only"
        run_synthesis "LIDC" 3
        
    elif [ "$1" = "emidec" ] || [ "$1" = "EMIDEC" ]; then
        echo "Processing EMIDEC dataset only"
        run_synthesis "EMIDEC" 2
        
    else
        echo "Processing all datasets"
        
        # LIDC: Generate 3 synthetic per normal (multi-peak lesions)
        run_synthesis "LIDC" 3
        
        # EMIDEC: Generate 2 synthetic per normal (multi-class lesions)
        run_synthesis "EMIDEC" 2
    fi
    
    echo ""
    echo "================================"
    echo "Synthesis Pipeline Complete!"
    echo "================================"
    
    # Show summary
    echo ""
    echo "Output directories:"
    echo "  LIDC:   $DATA_DIR/LIDC/synthetic_neuralsynth"
    echo "  EMIDEC: $DATA_DIR/EMIDEC/synthetic_neuralsynth"
    
    # Count generated files
    if [ -d "$DATA_DIR/LIDC/synthetic_neuralsynth" ]; then
        LIDC_COUNT=$(ls -1 $DATA_DIR/LIDC/synthetic_neuralsynth/*.npz 2>/dev/null | wc -l)
        echo "  LIDC synthetic cases: $LIDC_COUNT"
    fi
    
    if [ -d "$DATA_DIR/EMIDEC/synthetic_neuralsynth" ]; then
        EMIDEC_COUNT=$(ls -1 $DATA_DIR/EMIDEC/synthetic_neuralsynth/*.npz 2>/dev/null | wc -l)
        echo "  EMIDEC synthetic cases: $EMIDEC_COUNT"
    fi
}

# Run with arguments
main $@