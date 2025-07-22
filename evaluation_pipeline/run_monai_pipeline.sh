set -e # Exit immediately if a command exits with a non-zero status.

# Activate Conda Environment
source /usr/local/miniconda/etc/profile.d/conda.sh
conda activate lefusion

# --- Configuration ---
# CHOOSE YOUR MODEL: Uncomment the line for the model you want to use.

# Option 1: Use the official pre-trained model
LEFUSION_MODEL_PATH="LeFusion/LeFusion_Model/LIDC/lidc_pretrained.pt"

# Option 2: Use your self-trained model (uncomment below and comment above)
# LEFUSION_MODEL_PATH="LeFusion/LeFusion_Model/LIDC/model-21.pt"

# --- Other Configurations ---
SYNTHETIC_METHOD_NAME="LeFusion_H"
MODEL_NAME="SwinUNETR"
EXPERIMENT_NAME="${SYNTHETIC_METHOD_NAME}_P+N_prime"
PROJECT_ROOT_DIR=$(pwd)/..
REAL_DATA_DIR="datasets/LIDC_real"
SYNTHETIC_DATA_DIR="datasets/${SYNTHETIC_METHOD_NAME}_N_prime"
TRAINED_MODEL_DIR="trained_models/${EXPERIMENT_NAME}_${MODEL_NAME}"

echo "=========================================================="
echo "Starting Full Evaluation Pipeline for: ${EXPERIMENT_NAME}"
echo "Using LeFusion Model: ${LEFUSION_MODEL_PATH}"
echo "=========================================================="

# --- Step 0: Generate Synthetic Data ---
echo ">>> STEP 0: GENERATING SYNTHETIC DATA..."
mkdir -p "${SYNTHETIC_DATA_DIR}/imagesTr"
mkdir -p "${SYNTHETIC_DATA_DIR}/labelsTr"

(cd ${PROJECT_ROOT_DIR} && python LeFusion/inference/inference.py \
    data_type=lidc \
    model_path="${LEFUSION_MODEL_PATH}" \
    dataset_root_dir='data/LIDC/Normal/Image' \
    test_txt_dir='data/LIDC/Pathological/test.txt' \
    target_img_path="evaluation_pipeline/${SYNTHETIC_DATA_DIR}/imagesTr" \
    target_label_path="evaluation_pipeline/${SYNTHETIC_DATA_DIR}/labelsTr" \
    batch_size=1 \
    types=3 \
    diffusion_img_size=64 \
    diffusion_depth_size=32 \
    diffusion_num_channels=1 \
    cond_dim=16)

echo ">>> Synthetic data generated in ${SYNTHETIC_DATA_DIR}"

# --- Step 1: Create a temporary combined dataset for training ---
COMBINED_DATA_DIR="temp_combined_dataset"
echo ">>> STEP 1: Preparing combined dataset for training..."
rm -rf ${COMBINED_DATA_DIR}
cp -r ${REAL_DATA_DIR} ${COMBINED_DATA_DIR}
# This cp command needs to be fixed to handle the subdirectories (Image_1, etc.)
find "${SYNTHETIC_DATA_DIR}/imagesTr" -type f -name "*.nii.gz" -exec cp {} "${COMBINED_DATA_DIR}/imagesTr/" \;
find "${SYNTHETIC_DATA_DIR}/labelsTr" -type f -name "*.nii.gz" -exec cp {} "${COMBINED_DATA_DIR}/labelsTr/" \;
echo "Combined dataset ready."

# --- Step 2: Train the Segmentation Model using MONAI script ---
echo ">>> STEP 2: TRAINING SEGMENTATION MODEL..."
python monai_train.py \
    --data_dir "${COMBINED_DATA_DIR}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${TRAINED_MODEL_DIR}" \
    --max_epochs 200

# Clean up temporary data
rm -rf ${COMBINED_DATA_DIR}

echo "=========================================================="
echo "Training finished."
echo "Best model saved in ${TRAINED_MODEL_DIR}"
echo "=========================================================="