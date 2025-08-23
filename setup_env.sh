#!/bin/bash
# Source this file to set up the environment: source setup_env.sh

echo "Setting up persistent Python environment..."

# Set Python paths to use workspace directory
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
export PIP_TARGET=/workspace/diffing-toolkit/.local
export PATH=/workspace/diffing-toolkit/.local/bin:$PATH

# Create .local directory if it doesn't exist
mkdir -p /workspace/diffing-toolkit/.local

echo "Environment configured:"
echo "  PYTHONPATH includes: /workspace/diffing-toolkit/.local"
echo "  PIP will install to: /workspace/diffing-toolkit/.local"
echo "  PATH includes: /workspace/diffing-toolkit/.local/bin"
echo ""
echo "All pip installs will now go to the workspace directory (persistent)."