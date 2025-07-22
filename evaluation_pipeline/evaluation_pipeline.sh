# Activate Conda Environment
# source /usr/local/miniconda/etc/profile.d/conda.sh
# conda activate lefusion

# --- Configuration ---
# You can change these variables to test different methods
METHOD_NAME="LeFusion-H" # e.g., LeFusion, HandCrafted, etc.
MODEL_NAME="SwinUNETR"

# --- Static Paths ---
REAL_DATA_DIR="datasets/LIDC_real"
SYNTHETIC_DATA_DIR="datasets/${METHOD_NAME}_N_prime"
EXPERIMENT_NAME="${METHOD_NAME}_P+N_prime"
TRAINED_MODEL_DIR="trained_models/${EXPERIMENT_NAME}_${MODEL_NAME}"
RESULTS_CSV="evaluation_results.csv"

echo "=========================================================="
echo "STARTING FULL PIPELINE"
echo "Method: ${METHOD_NAME} | Model: ${MODEL_NAME}"
echo "=========================================================="

# --- Step 1: Generate Synthetic Data (if it doesn't exist) ---
echo ">>> STEP 1: GENERATING SYNTHETIC DATA..."

# Create the output directories for the synthetic data
mkdir -p "${SYNTHETIC_DATA_DIR}/imagesTr"
mkdir -p "${SYNTHETIC_DATA_DIR}/labelsTr"

# NOTE: This assumes you have already run the prepare_real_dataset.py script
# The LeFusion inference script generates new images based on the 'Normal' data
# and saves the corresponding masks. We need to rename them for our pipeline.

# Run LeFusion inference to create synthetic data
# We run this from the main LeFusion directory
(
  cd .. && python LeFusion/inference/inference.py \
    data_type=lidc \
    model_path='LeFusion/LeFusion_Model/LIDC/lidc.pt' \
    dataset_root_dir='data/LIDC/Normal/Image' \
    test_txt_dir='data/LIDC/Pathological/test.txt' \
    target_img_path="evaluation_pipeline/${SYNTHETIC_DATA_DIR}/temp_images" \
    target_label_path="evaluation_pipeline/${SYNTHETIC_DATA_DIR}/temp_labels" \
    batch_size=4 \
    types=3
)