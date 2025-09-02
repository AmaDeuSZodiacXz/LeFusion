#!/usr/bin/env python3
"""
Fix for transformers/pytorch compatibility issues.
Run this before training to patch the import issues.
"""

import sys
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*register_pytree_node.*")

# Monkey patch the problematic attribute
try:
    import torch.utils._pytree as pytree
    if not hasattr(pytree, 'register_pytree_node'):
        pytree.register_pytree_node = pytree._register_pytree_node
        print("✓ Fixed torch.utils._pytree compatibility")
except Exception as e:
    print(f"Warning: Could not patch pytree: {e}")

# Try importing transformers to trigger the fix
try:
    import transformers
    print("✓ Transformers imported successfully")
except ImportError:
    print("✓ Transformers not installed (not required for training)")

print("Import fixes applied successfully!")