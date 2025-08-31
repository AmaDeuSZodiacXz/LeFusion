#!/bin/bash

# Comprehensive Paper Evaluation Pipeline for LeFusion
# Reproduces the exact evaluation table from the paper

# Function to print messages (no logfile)
print_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

set -e

print_message "=========================================================="
print_message "LEFUSION COMPREHENSIVE PAPER EVALUATION PIPELINE"
print_message "Reproducing the exact evaluation table from the paper"
print_message "=========================================================="

# Configuration
METHODS=("baseline" "lefusion" "lefusion_h" "lefusion_h_diffmask")
MODEL_TYPES=("pretrained" "from_scratch")
SEGMENTATION_MODELS=("nnunet" "swinunetr")

# Check if we're in the evaluation_pipeline directory
if [ ! -f "run_comprehensive_paper_evaluation.py" ]; then
    print_message "Error: Please run this script from the evaluation_pipeline directory"
    exit 1
fi

# Function to check if resume is needed
check_resume() {
    if [ -d "paper_experiments" ] && [ "$(ls -A paper_experiments 2>/dev/null)" ]; then
        print_message "Found existing experiments directory. Checking progress..."
        python run_paper_evaluation_resume.py --check_only
        echo ""
        read -p "Do you want to resume from existing progress? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_message "Resuming from existing progress..."
            return 0  # Resume
        else
            print_message "Starting fresh pipeline..."
            return 1  # Start fresh
        fi
    fi
    return 1  # Start fresh
}

# Function to run full pipeline
run_full_pipeline() {
    print_message ">>> Running full paper evaluation pipeline..."
    python run_comprehensive_paper_evaluation.py \
        --methods "${METHODS[@]}" \
        --model_types "${MODEL_TYPES[@]}" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Function to run resume pipeline
run_resume_pipeline() {
    print_message ">>> Resuming paper evaluation pipeline..."
    python run_paper_evaluation_resume.py \
        --methods "${METHODS[@]}" \
        --model_types "${MODEL_TYPES[@]}" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Function to run specific method
run_specific_method() {
    local method=$1
    print_message ">>> Running specific method: $method"
    python run_comprehensive_paper_evaluation.py \
        --methods "$method" \
        --model_types "${MODEL_TYPES[@]}" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Function to run specific model type
run_specific_model_type() {
    local model_type=$1
    print_message ">>> Running specific model type: $model_type"
    python run_comprehensive_paper_evaluation.py \
        --methods "${METHODS[@]}" \
        --model_types "$model_type" \
        --segmentation_models "${SEGMENTATION_MODELS[@]}"
}

# Main execution
if [ "$1" = "resume" ]; then
    print_message "Resume mode selected"
    run_resume_pipeline
elif [ "$1" = "check" ]; then
    print_message "Check mode selected"
    python run_paper_evaluation_resume.py --check_only
elif [ "$1" = "method" ] && [ -n "$2" ]; then
    print_message "Specific method mode selected: $2"
    run_specific_method "$2"
elif [ "$1" = "model_type" ] && [ -n "$2" ]; then
    print_message "Specific model type mode selected: $2"
    run_specific_model_type "$2"
elif [ "$1" = "fresh" ]; then
    print_message "Fresh start mode selected"
    run_full_pipeline
else
    # Interactive mode
    print_message "Available options:"
    print_message "  ./run_paper_pipeline.sh              # Interactive mode"
    print_message "  ./run_paper_pipeline.sh resume       # Resume from existing progress"
    print_message "  ./run_paper_pipeline.sh check        # Check existing progress"
    print_message "  ./run_paper_pipeline.sh fresh        # Start fresh (ignore existing)"
    print_message "  ./run_paper_pipeline.sh method <method>     # Run specific method"
    print_message "  ./run_paper_pipeline.sh model_type <type>   # Run specific model type"
    print_message ""
    print_message "Methods: ${METHODS[*]}"
    print_message "Model Types: ${MODEL_TYPES[*]}"
    print_message "Segmentation Models: ${SEGMENTATION_MODELS[*]}"
    print_message ""
    
    if check_resume; then
        run_resume_pipeline
    else
        run_full_pipeline
    fi
fi

print_message "=========================================================="
print_message "PAPER EVALUATION PIPELINE COMPLETED"
print_message "Check the generated CSV files for results"
print_message "=========================================================="

# Display results if available
if [ -f "comprehensive_paper_results.csv" ]; then
    echo ""
    echo "Results Summary:"
    echo "==============="
    tail -n 5 comprehensive_paper_results.csv
fi 