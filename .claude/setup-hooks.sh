#!/bin/bash
# Setup script to make all hooks executable

echo "Setting up Claude Code hooks..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOOKS_DIR="$SCRIPT_DIR/hooks"

# Make all Python files in hooks directory executable
if [ -d "$HOOKS_DIR" ]; then
    echo "Making hooks executable in: $HOOKS_DIR"
    chmod +x "$HOOKS_DIR"/*.py
    echo "✅ All hooks are now executable!"
else
    echo "❌ Hooks directory not found at: $HOOKS_DIR"
    exit 1
fi

# Verify permissions
echo -e "\nHook permissions:"
ls -la "$HOOKS_DIR"/*.py