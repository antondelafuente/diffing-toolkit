#!/bin/bash
# Safe pip wrapper that prevents reinstalling critical packages

# List of packages that should NEVER be reinstalled
PROTECTED_PACKAGES="torch torchvision torchaudio nvidia cuda cudnn triton"

# Check if any protected package is being installed
for arg in "$@"; do
    for protected in $PROTECTED_PACKAGES; do
        if [[ "$arg" == *"$protected"* ]]; then
            echo "❌ ERROR: Attempting to install protected package: $protected"
            echo "These packages are already installed system-wide and should not be reinstalled."
            echo "Use --no-deps flag or install other packages only."
            exit 1
        fi
    done
done

# Set target to workspace and run pip
export PIP_TARGET=/workspace/diffing-toolkit/.local
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH

echo "✅ Running safe pip install to workspace directory..."
pip "$@"