#!/usr/bin/env python3
"""
Simplified training script that avoids the transformers compatibility issue.
Run this instead of train_lidc.py from the synthetic_training directory.
"""

import os
import sys
from pathlib import Path

# Fix the compatibility issue before any other imports
import torch.utils._pytree as pytree
if not hasattr(pytree, 'register_pytree_node'):
    pytree.register_pytree_node = pytree._register_pytree_node

# Now we can import everything else
os.chdir(Path(__file__).parent / 'synthetic_training')
sys.path.insert(0, str(Path(__file__).parent / 'synthetic_training'))

# Import and run the main training script
from train_lidc import main

if __name__ == "__main__":
    main()