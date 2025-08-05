# Claude Code Setup Instructions

## Quick Setup

After copying the `.claude` directory to a new project, run:

```bash
cd .claude
./setup-hooks.sh
```

Or if you have make installed:

```bash
cd .claude
make setup
```

## Manual Setup

If the scripts don't work, manually set permissions:

```bash
chmod +x .claude/hooks/*.py
chmod +x .claude/setup-hooks.sh
```

## Using with Git

To preserve permissions when committing to Git:

```bash
# Add files with executable permission
git add --chmod=+x .claude/hooks/*.py
git add --chmod=+x .claude/setup-hooks.sh
```

## Copying to New Projects

### Option 1: Using rsync (preserves permissions)
```bash
rsync -av --include=".*" source/project/.claude/ new/project/.claude/
```

### Option 2: Using tar (preserves permissions)
```bash
# Create archive
tar -czf claude-setup.tar.gz .claude/

# Extract in new project
cd /path/to/new/project
tar -xzf /path/to/claude-setup.tar.gz
```

### Option 3: Using cp with permissions
```bash
cp -rp source/project/.claude new/project/
```

## Troubleshooting

If hooks aren't running:

1. Check permissions: `ls -la .claude/hooks/`
2. Look for `x` in permissions (e.g., `-rwxr-xr-x`)
3. Run setup script: `.claude/setup-hooks.sh`
4. Check Claude Code logs: `claude --debug`