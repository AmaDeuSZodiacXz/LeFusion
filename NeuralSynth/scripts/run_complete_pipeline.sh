#!/bin/bash

# ============================================
# NeuralSynth Complete Pipeline
# Following LeFusion evaluation_training structure
# ============================================

echo "╔══════════════════════════════════════════╗"
echo "║     NeuralSynth Complete Pipeline        ║"
echo "║   Following evaluation_training structure ║"
echo "╚══════════════════════════════════════════╝"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set directories using relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEURALSYNTH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$NEURALSYNTH_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
EVAL_TRAINING_DIR="$PROJECT_ROOT/evaluation_training"

# Default settings
DATASET="lidc"
METHOD="neuralsynth"
SEG_MODEL="nnunet"
SKIP_SYNTHETIC=false
SKIP_TRAINING=false
SKIP_EVALUATION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --seg-model)
            SEG_MODEL="$2"
            shift 2
            ;;
        --skip-synthetic)
            SKIP_SYNTHETIC=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset [lidc|emidec]     Dataset to use (default: lidc)"
            echo "  --method METHOD              Method name (default: neuralsynth)"
            echo "  --seg-model [nnunet|swinunetr] Segmentation model (default: nnunet)"
            echo "  --skip-synthetic             Skip synthetic data generation"
            echo "  --skip-training              Skip model training"
            echo "  --skip-evaluation            Skip evaluation"
            echo "  --help                       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored status
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "$BLUE" "\n📋 Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_status "$RED" "✗ Python 3 not found"
        exit 1
    fi
    
    # Check PyTorch
    python3 -c "import torch" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_status "$YELLOW" "⚠ PyTorch not installed"
        exit 1
    fi
    
    # Check data directory
    if [ ! -d "$DATA_DIR/$DATASET" ]; then
        print_status "$RED" "✗ Dataset not found: $DATA_DIR/$DATASET"
        exit 1
    fi
    
    print_status "$GREEN" "✓ Prerequisites checked"
}

# Step 1: Generate Synthetic Data
generate_synthetic_data() {
    if [ "$SKIP_SYNTHETIC" = true ]; then
        print_status "$YELLOW" "⏭ Skipping synthetic data generation"
        return
    fi
    
    print_status "$BLUE" "\n🔬 Step 1/3: Generating Synthetic Data"
    print_status "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd "$NEURALSYNTH_DIR"
    
    # Check if synthetic data already exists
    SYNTH_DIR="$NEURALSYNTH_DIR/synthetic_data/$DATASET/$METHOD"
    if [ -d "$SYNTH_DIR/P_N_prime" ]; then
        print_status "$YELLOW" "⚠ Synthetic data already exists at $SYNTH_DIR"
        read -p "Regenerate? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi
    
    # Generate synthetic data
    python3 synthetic_generation/generate_synthetic.py \
        --dataset "$DATASET" \
        --method "$METHOD" \
        --data-dir "$DATA_DIR" \
        --output-dir "$SYNTH_DIR" \
        --combinations P_P_prime P_N_prime P_N_double_prime
    
    if [ $? -eq 0 ]; then
        print_status "$GREEN" "✓ Synthetic data generated successfully"
        
        # Count generated files
        if [ -d "$SYNTH_DIR/P_N_prime" ]; then
            COUNT=$(ls -1 "$SYNTH_DIR/P_N_prime"/*_image.nii.gz 2>/dev/null | wc -l)
            print_status "$GREEN" "  Generated $COUNT synthetic samples in P_N_prime"
        fi
    else
        print_status "$RED" "✗ Synthetic data generation failed"
        exit 1
    fi
}

# Step 2: Train Segmentation Models
train_segmentation_models() {
    if [ "$SKIP_TRAINING" = true ]; then
        print_status "$YELLOW" "⏭ Skipping model training"
        return
    fi
    
    print_status "$BLUE" "\n🎯 Step 2/3: Training Segmentation Models"
    print_status "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd "$NEURALSYNTH_DIR"
    
    # Define data combinations to train
    COMBINATIONS=("P" "P_P_prime" "P_N_prime" "P_P_prime_N_double_prime")
    
    for COMBINATION in "${COMBINATIONS[@]}"; do
        print_status "$BLUE" "\nTraining with combination: $COMBINATION"
        
        python3 training/train_segmentation.py \
            --dataset "$DATASET" \
            --method "$METHOD" \
            --combination "$COMBINATION" \
            --seg-model "$SEG_MODEL"
        
        if [ $? -eq 0 ]; then
            print_status "$GREEN" "✓ Training completed for $COMBINATION"
        else
            print_status "$YELLOW" "⚠ Training failed for $COMBINATION"
        fi
    done
}

# Step 3: Evaluate Models
evaluate_models() {
    if [ "$SKIP_EVALUATION" = true ]; then
        print_status "$YELLOW" "⏭ Skipping evaluation"
        return
    fi
    
    print_status "$BLUE" "\n📊 Step 3/3: Evaluating Models"
    print_status "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd "$NEURALSYNTH_DIR"
    
    python3 evaluation/evaluate_models.py \
        --dataset "$DATASET" \
        --method "$METHOD" \
        --seg-model "$SEG_MODEL" \
        --compare-paper \
        --use-best-checkpoint
    
    if [ $? -eq 0 ]; then
        print_status "$GREEN" "✓ Evaluation completed successfully"
        
        # Show results
        RESULTS_FILE="$NEURALSYNTH_DIR/evaluation_results/${DATASET}_results.json"
        if [ -f "$RESULTS_FILE" ]; then
            print_status "$BLUE" "\n📈 Results Summary:"
            python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
    for method, metrics in results.items():
        if 'dice' in metrics:
            print(f'  {method}: DICE={metrics[\"dice\"]:.4f}, NSD={metrics.get(\"nsd\", 0):.4f}')
"
        fi
    else
        print_status "$RED" "✗ Evaluation failed"
        exit 1
    fi
}

# Function to show comparison with LeFusion
show_comparison() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║        Pipeline Complete! 🎉             ║"
    echo "╚══════════════════════════════════════════╝"
    
    print_status "$GREEN" "\n📊 Comparison with LeFusion:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ "$DATASET" = "lidc" ]; then
        echo "Dataset: LIDC-IDRI"
        echo ""
        echo "Method                    | DICE  | NSD   |"
        echo "--------------------------|-------|-------|"
        echo "LeFusion Baseline         | 78.26 | 88.90 |"
        echo "LeFusion                  | 78.77 | 89.25 |"
        echo "LeFusion-H                | 80.62 | 90.90 |"
        echo "LeFusion-H+DiffMask       | 83.44 | 93.35 |"
        echo "NeuralSynth (Target)      | 89.20 | 95.40 |"
    else
        echo "Dataset: EMIDEC"
        echo ""
        echo "Method                    | MI    | PMO   |"
        echo "--------------------------|-------|-------|"
        echo "LeFusion Baseline         | 68.61 | 36.32 |"
        echo "LeFusion                  | 69.88 | 34.79 |"
        echo "LeFusion-H                | 69.95 | 38.01 |"
        echo "LeFusion-H+DiffMask       | 71.28 | 43.41 |"
        echo "NeuralSynth (Target)      | 75.00 | 48.00 |"
    fi
    
    # Show output locations
    print_status "$BLUE" "\n📁 Output Locations:"
    echo "━━━━━━━━━━━━━━━━━━━━"
    echo "Synthetic Data: $NEURALSYNTH_DIR/synthetic_data/$DATASET/"
    echo "Trained Models: $NEURALSYNTH_DIR/trained_models/$DATASET/"
    echo "Results: $NEURALSYNTH_DIR/evaluation_results/"
}

# Main execution
main() {
    print_status "$YELLOW" "\n🚀 Starting NeuralSynth Pipeline"
    print_status "$YELLOW" "Dataset: $DATASET | Method: $METHOD | Model: $SEG_MODEL"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Run pipeline steps
    generate_synthetic_data
    train_segmentation_models
    evaluate_models
    
    # Show comparison
    show_comparison
}

# Make script executable
chmod +x "$0"

# Run main function
main