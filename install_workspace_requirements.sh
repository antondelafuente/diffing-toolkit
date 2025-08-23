#!/bin/bash
# Install all requirements to workspace without touching PyTorch/CUDA
# Run this after pod restart to restore the environment

echo "Installing workspace requirements..."
echo "This will NOT touch PyTorch, CUDA, or other system packages"

# Set up environment
export PIP_TARGET=/workspace/diffing-toolkit/.local
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
mkdir -p /workspace/diffing-toolkit/.local

# Install packages without dependencies to avoid torch reinstallation
# Note: This assumes torch and other system packages are already installed
pip install --no-deps -r requirements-workspace.txt

# Special case: dictionary_learning from git
if [ ! -d "/workspace/diffing-toolkit/.local/dictionary_learning" ]; then
    echo "Installing dictionary_learning..."
    cd /tmp
    git clone https://github.com/science-of-finetuning/dictionary_learning.git
    cp -r dictionary_learning/dictionary_learning /workspace/diffing-toolkit/.local/
    rm -rf dictionary_learning
    cd -
fi

echo "Installation complete!"
echo "Remember to set these environment variables:"
echo "  export PIP_TARGET=/workspace/diffing-toolkit/.local"
echo "  export PYTHONPATH=/workspace/diffing-toolkit/.local:\$PYTHONPATH"
echo ""
echo "Or run: source /workspace/startup.sh"