#!/bin/bash

echo "=== RunPod Startup Script ==="
echo "Starting initialization..."

# Determine current user
CURRENT_USER=$(whoami)
echo "Running as: $CURRENT_USER"

# If running as root, set up anton user
if [ "$CURRENT_USER" = "root" ]; then
    # Install tmux if not already installed
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux
    echo "tmux installed successfully"
    
    # Create anton user if doesn't exist
    echo "Checking for anton user..."
    if id -u anton >/dev/null 2>&1; then
        echo "User anton already exists"
    else
        echo "Creating user anton with sudo privileges..."
        useradd -m -s /bin/bash anton && echo "anton ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
        echo "User anton created successfully"
    fi
    
    # Setup NVM for anton user (append only if not already there)
    if ! grep -q "NVM_DIR=/workspace/.nvm" /home/anton/.bashrc 2>/dev/null; then
        echo "Setting up NVM for anton user..."
        printf '\nexport NVM_DIR=/workspace/.nvm\n. "$NVM_DIR/nvm.sh"\n' >> /home/anton/.bashrc
    fi
    
    # Setup Git configuration persistence
    echo "Setting up persistent Git configuration..."
    ln -sf /workspace/.gitconfig /home/anton/.gitconfig 2>/dev/null || true
    touch /workspace/.git-credentials
    chmod 600 /workspace/.git-credentials
    su - anton -c 'git config --global credential.helper "store --file=/workspace/.git-credentials"' 2>/dev/null || true
    echo "Git persistence configured"
    
    # Setup persistent Claude Code storage
    if [ ! -d "/workspace/.claude-persistent" ]; then
        echo "Creating initial Claude persistent storage..."
        # If anton has existing Claude data, preserve it
        if [ -d "/home/anton/.claude" ]; then
            echo "Migrating existing Claude data to persistent storage..."
            cp -r /home/anton/.claude /workspace/.claude-persistent
            [ -f "/home/anton/.claude.json" ] && cp /home/anton/.claude.json /workspace/.claude-persistent.json
            [ -f "/home/anton/.claude.json.backup" ] && cp /home/anton/.claude.json.backup /workspace/.claude-persistent.json.backup
        else
            # Create fresh directories
            mkdir -p /workspace/.claude-persistent
            touch /workspace/.claude-persistent.json
            touch /workspace/.claude-persistent.json.backup
        fi
        # Set proper permissions
        chown -R anton:anton /workspace/.claude-persistent*
        chmod -R 777 /workspace/.claude-persistent*
        echo "Claude persistent storage created"
    fi
    
    # Now link the persistent storage
    echo "Linking Claude persistent storage..."
    rm -rf /home/anton/.claude /home/anton/.claude.json /home/anton/.claude.json.backup 2>/dev/null || true
    ln -sf /workspace/.claude-persistent /home/anton/.claude 2>/dev/null || true
    ln -sf /workspace/.claude-persistent.json /home/anton/.claude.json 2>/dev/null || true
    ln -sf /workspace/.claude-persistent.json.backup /home/anton/.claude.json.backup 2>/dev/null || true
    echo "Claude persistence linked successfully"
    
    # Setup Python environment for persistent package installation (only if not already configured)
    if ! grep -q "PYTHONPATH=/workspace/diffing-toolkit/.local" /home/anton/.bashrc 2>/dev/null; then
        echo "Setting up persistent Python environment..."
        mkdir -p /workspace/diffing-toolkit/.local
        cat >> /home/anton/.bashrc << 'EOF'

# Persistent Python environment for diffing-toolkit
export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
export PIP_TARGET=/workspace/diffing-toolkit/.local
export PATH=/workspace/diffing-toolkit/.local/bin:$PATH

# HuggingFace token for accessing gated models (e.g., Gemma)
# Load from file if it exists (not committed to git)
if [ -f "/workspace/.hf_token" ]; then
    export HF_TOKEN=$(cat /workspace/.hf_token)
fi

# Set HuggingFace cache to network volume to avoid filling container storage
export HF_HOME=/workspace/.cache/huggingface

# Safety alias to prevent torch/cuda reinstalls
alias pip='echo "⚠️  Use pip-safe or pip --no-deps to avoid reinstalling torch/cuda!"; echo "Run: pip-safe <args> for safe installation"; false'
alias pip-safe='/workspace/diffing-toolkit/pip_safe.sh'
alias pip-force='/usr/bin/pip'  # Emergency bypass if really needed
EOF
        echo "Python environment configured with safety measures"
    else
        echo "Python environment already configured"
    fi
    
    # Switch to anton and continue
    echo "Switching to anton user..."
    TARGET_USER="anton"
elif [ "$CURRENT_USER" = "anton" ]; then
    # Already running as anton - skip user switching
    echo "Already running as anton user - skipping user setup"
    TARGET_USER="anton"
    
    # Still ensure Python environment is set up for current session
    export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
    export PIP_TARGET=/workspace/diffing-toolkit/.local
    export PATH=/workspace/diffing-toolkit/.local/bin:$PATH
    
    # Load HF token if exists
    if [ -f "/workspace/.hf_token" ]; then
        export HF_TOKEN=$(cat /workspace/.hf_token)
    fi
    
    # Set HuggingFace cache to network volume
    export HF_HOME=/workspace/.cache/huggingface
else
    # Running as another non-root user
    echo "Running as non-root user: $CURRENT_USER"
    TARGET_USER="$CURRENT_USER"
    
    # Still ensure Python environment is set up for current session
    export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH
    export PIP_TARGET=/workspace/diffing-toolkit/.local
    export PATH=/workspace/diffing-toolkit/.local/bin:$PATH
    
    # Load HF token if exists
    if [ -f "/workspace/.hf_token" ]; then
        export HF_TOKEN=$(cat /workspace/.hf_token)
    fi
    
    # Set HuggingFace cache to network volume
    export HF_HOME=/workspace/.cache/huggingface
fi

# Function to launch the environment
launch_environment() {
    cd /workspace/diffing-toolkit
    
    # Just set up environment, don't launch Claude
    echo "Environment setup complete!"
    echo "Python packages: /workspace/diffing-toolkit/.local"
    echo "HF token loaded: $([ -f /workspace/.hf_token ] && echo 'Yes' || echo 'No')"
}

# Execute the appropriate action based on current user
if [ "$CURRENT_USER" = "root" ] && [ "$TARGET_USER" = "anton" ]; then
    # Switch to anton and run the launch function with NVM loaded
    exec su - anton -c 'export NVM_DIR=/workspace/.nvm; . "$NVM_DIR/nvm.sh"; export PYTHONPATH=/workspace/diffing-toolkit/.local:$PYTHONPATH; export PIP_TARGET=/workspace/diffing-toolkit/.local; export PATH=/workspace/diffing-toolkit/.local/bin:$PATH; [ -f /workspace/.hf_token ] && export HF_TOKEN=$(cat /workspace/.hf_token); export HF_HOME=/workspace/.cache/huggingface; cd /workspace/diffing-toolkit && echo "Environment setup complete!" && echo "Python packages: /workspace/diffing-toolkit/.local" && echo "HF token loaded: $([ -f /workspace/.hf_token ] && echo Yes || echo No)" && exec bash'
else
    # Already the right user, just launch
    launch_environment
fi