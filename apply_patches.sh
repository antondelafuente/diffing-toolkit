#!/bin/bash
# Apply patches to fix known issues in dependencies

echo "Applying patches to dictionary_learning..."

# Apply the cache.py fix for .value attribute error
if [ -f "/workspace/diffing-toolkit/.local/dictionary_learning/cache.py" ]; then
    echo "Applying dictionary_learning cache.py fix..."
    patch -p1 -d /workspace/diffing-toolkit/.local < patches/dictionary_learning_cache_fix.patch
    echo "Patch applied successfully!"
else
    echo "Warning: dictionary_learning not found in .local directory"
    echo "Run install_workspace_requirements.sh first"
fi

echo "All patches applied!"