#!/bin/bash

# Comprehensive Paper Evaluation Pipeline for LeFusion
# Reproduces the exact evaluation table from the paper

set -e

echo "=========================================================="
echo "LEFUSION COMPREHENSIVE PAPER EVALUATION PIPELINE"
echo "Reproducing the exact evaluation table from the paper"
echo "=========================================================="

# Configuration
METHODS=("baseline" "lefusion" "lefusion_h" "lefusion_h_diffmask")
MODEL_TYPES=("pretrained" "from_scratch")
SEGMENTATION_MODELS=("nnUNet" "SwinUNETR")

# Check if we're in the evaluation_pipeline directory
if [ ! -f "run_comprehensive_paper_evaluation.py" ]; then
    echo "Error: Please run this script from the evaluation_pipeline directory"
    exit 1
fi

# Function to check if resume is needed
check_resume() {
    if [ -d "paper_experiments" ] && [ "$(ls -A paper_experiments 2>/dev/null)" ]; then
        echo "Found existing experiments directory. Checking progress..."
        python run_paper_evaluation_resume.py --check_only
        echo ""
        read -p "Do you want to resume from existing progress? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            return 0  # Resume
        else
            echo "Starting fresh pipeline..."
            return 1  # Start fresh
        fi
    fi
    return 1  # Start fresh
}

# Function to run full pipeline
run_full_pipeline() {
    echo ">>> Running full paper evaluation pipeline..."
    python run_comprehensive_paper_evaluation.py \
        --methods "${METHODS[@]}" \
        --model_types "${MODEL_TYPES[@]}" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Function to run resume pipeline
run_resume_pipeline() {
    echo ">>> Resuming paper evaluation pipeline..."
    python run_paper_evaluation_resume.py \
        --methods "${METHODS[@]}" \
        --model_types "${MODEL_TYPES[@]}" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Function to run specific method
run_specific_method() {
    local method=$1
    echo ">>> Running specific method: $method"
    python run_comprehensive_paper_evaluation.py \
        --methods "$method" \
        --model_types "${MODEL_TYPES[@]}" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Function to run specific model type
run_specific_model_type() {
    local model_type=$1
    echo ">>> Running specific model type: $model_type"
    python run_comprehensive_paper_evaluation.py \
        --methods "${METHODS[@]}" \
        --model_types "$model_type" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Main execution
if [ "$1" = "resume" ]; then
    echo "Resume mode selected"
    run_resume_pipeline
elif [ "$1" = "check" ]; then
    echo "Check mode selected"
    python run_paper_evaluation_resume.py --check_only
elif [ "$1" = "method" ] && [ -n "$2" ]; then
    echo "Specific method mode selected: $2"
    run_specific_method "$2"
elif [ "$1" = "model_type" ] && [ -n "$2" ]; then
    echo "Specific model type mode selected: $2"
    run_specific_model_type "$2"
elif [ "$1" = "fresh" ]; then
    echo "Fresh start mode selected"
    run_full_pipeline
else
    # Interactive mode
    echo "Available options:"
    echo "  ./run_paper_pipeline.sh              # Interactive mode"
    echo "  ./run_paper_pipeline.sh resume       # Resume from existing progress"
    echo "  ./run_paper_pipeline.sh check        # Check existing progress"
    echo "  ./run_paper_pipeline.sh fresh        # Start fresh (ignore existing)"
    echo "  ./run_paper_pipeline.sh method <method>     # Run specific method"
    echo "  ./run_paper_pipeline.sh model_type <type>   # Run specific model type"
    echo ""
    echo "Methods: ${METHODS[*]}"
    echo "Model Types: ${MODEL_TYPES[*]}"
    echo "Segmentation Models: ${SEGMENTATION_MODELS[*]}"
    echo ""
    
    if check_resume; then
        run_resume_pipeline
    else
        run_full_pipeline
    fi
fi

echo "=========================================================="
echo "PAPER EVALUATION PIPELINE COMPLETED"
echo "Check the generated CSV files for results"
echo "=========================================================="

# Display results if available
if [ -f "comprehensive_paper_results.csv" ]; then
    echo ""
    echo "Results Summary:"
    echo "==============="
    tail -n 5 comprehensive_paper_results.csv
fi 