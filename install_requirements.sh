#!/bin/bash
# Safe installation script for requirements.txt

echo "Installing requirements safely (excluding torch/cuda)..."

# Set environment
export PIP_TARGET=/workspace/diffing-toolkit/.local
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH

# Read requirements and filter out dangerous packages
SAFE_PACKAGES=""
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue
    
    # Skip torch and cuda related packages
    [[ "$line" =~ ^torch ]] && echo "⏭️  Skipping: $line (already installed)" && continue
    [[ "$line" =~ nvidia ]] && echo "⏭️  Skipping: $line (already installed)" && continue
    [[ "$line" =~ cuda ]] && echo "⏭️  Skipping: $line (already installed)" && continue
    
    SAFE_PACKAGES="$SAFE_PACKAGES $line"
done < requirements.txt

echo "Installing: $SAFE_PACKAGES"
pip install --no-deps $SAFE_PACKAGES

echo "✅ Installation complete. Torch/CUDA packages preserved."