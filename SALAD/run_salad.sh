#!/bin/bash

# SALAD Main Launcher Script

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           SALAD Launcher               ${NC}"
echo -e "${BLUE}========================================${NC}"

if [ "$1" == "inference" ]; then
    echo -e "${YELLOW}Running inference pipeline...${NC}"
    cd inference
    python inference_pipeline.py "${@:2}"
elif [ "$1" == "train" ]; then
    echo -e "${YELLOW}Starting training...${NC}"
    cd training
    if [ "$2" == "lidc" ]; then
        ./scripts/train_lidc.sh
    elif [ "$2" == "emidec" ]; then
        ./scripts/train_emidec.sh
    else
        echo -e "${RED}Please specify dataset: lidc or emidec${NC}"
    fi
elif [ "$1" == "quick" ]; then
    echo -e "${YELLOW}Running quick inference...${NC}"
    cd inference
    python quick_inference.py "${@:2}"
else
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./run_salad.sh inference [options]  - Run inference"
    echo "  ./run_salad.sh train [lidc|emidec]  - Train model"
    echo "  ./run_salad.sh quick [checkpoint]   - Quick test"
fi
