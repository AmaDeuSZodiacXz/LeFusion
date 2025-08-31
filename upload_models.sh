#!/bin/bash

# Upload LeFusion Models to Hugging Face
# Usage: bash upload_models.sh <repo_id> [options]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  LeFusion Model Upload to Hugging Face${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if repo_id is provided
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Repository ID is required${NC}"
    echo
    echo "Usage: bash upload_models.sh <repo_id> [options]"
    echo
    echo "Arguments:"
    echo "  repo_id          Hugging Face repository ID (e.g., username/lefusion-models)"
    echo
    echo "Options:"
    echo "  --token TOKEN    Hugging Face API token (or set HF_TOKEN env var)"
    echo "  --private        Make repository private"
    echo "  --include-trained Include trained segmentation models"
    echo "  --dry-run        Show what would be uploaded without uploading"
    echo
    echo "Examples:"
    echo "  bash upload_models.sh myusername/lefusion-models"
    echo "  bash upload_models.sh myusername/lefusion-models --token hf_xxxxx --private"
    echo "  bash upload_models.sh myusername/lefusion-models --include-trained"
    echo "  bash upload_models.sh myusername/lefusion-models --dry-run"
    exit 1
fi

REPO_ID=$1
shift  # Remove first argument

# Check if huggingface_hub is installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}Installing huggingface-hub...${NC}"
    pip install huggingface-hub
fi

# Check if HF token is available
if [ -z "$HF_TOKEN" ] && [[ ! " $@ " =~ " --token " ]]; then
    echo -e "${YELLOW}Warning: No HF_TOKEN environment variable found and no --token provided${NC}"
    echo -e "${YELLOW}You may need to login with: huggingface-cli login${NC}"
    echo
    read -p "Do you want to login now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    fi
fi

# List model directories to be uploaded
echo -e "${GREEN}Model directories to upload:${NC}"
echo "  1. /Users/skb/Documents/LeFusion/DiffMask/DiffMask_Model"
echo "  2. /Users/skb/Documents/LeFusion/LeFusion/LeFusion_Model"

if [[ " $@ " =~ " --include-trained " ]]; then
    echo "  3. /Users/skb/Documents/LeFusion/evaluation_pipeline_v2/trained_models"
fi

echo

# Count model files
echo -e "${YELLOW}Scanning for model files...${NC}"
TOTAL_FILES=0

for dir in "/Users/skb/Documents/LeFusion/DiffMask/DiffMask_Model" \
           "/Users/skb/Documents/LeFusion/LeFusion/LeFusion_Model"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) 2>/dev/null | wc -l)
        TOTAL_FILES=$((TOTAL_FILES + count))
        echo "  Found $count model files in $(basename $(dirname $dir))/$(basename $dir)"
    fi
done

if [[ " $@ " =~ " --include-trained " ]]; then
    dir="/Users/skb/Documents/LeFusion/evaluation_pipeline_v2/trained_models"
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) 2>/dev/null | wc -l)
        TOTAL_FILES=$((TOTAL_FILES + count))
        echo "  Found $count model files in trained_models/"
    fi
fi

echo -e "${GREEN}Total model files found: $TOTAL_FILES${NC}"
echo

# Confirm before upload (unless dry-run)
if [[ ! " $@ " =~ " --dry-run " ]]; then
    echo -e "${YELLOW}This will upload $TOTAL_FILES model files to:${NC}"
    echo -e "${BLUE}  https://huggingface.co/$REPO_ID${NC}"
    echo
    read -p "Do you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Upload cancelled${NC}"
        exit 1
    fi
fi

# Run the upload script
echo -e "${GREEN}Starting upload...${NC}"
echo

python /Users/skb/Documents/LeFusion/upload_to_huggingface.py "$REPO_ID" "$@"

# Check if upload was successful
if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Upload completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    if [[ ! " $@ " =~ " --dry-run " ]]; then
        echo
        echo -e "${BLUE}View your models at:${NC}"
        echo -e "${BLUE}  https://huggingface.co/$REPO_ID${NC}"
        echo
        echo -e "${YELLOW}To use these models:${NC}"
        echo "  1. Install: pip install huggingface-hub"
        echo "  2. Download: huggingface-cli download $REPO_ID --local-dir ./models"
        echo "  3. Or use in Python:"
        echo "     from huggingface_hub import hf_hub_download"
        echo "     model_path = hf_hub_download(repo_id='$REPO_ID', filename='LeFusion/LIDC/lidc.pt')"
    fi
else
    echo
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Upload failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Please check the error messages above${NC}"
    exit 1
fi