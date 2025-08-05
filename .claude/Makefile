# Makefile for Claude Code setup

.PHONY: setup permissions help

help:
	@echo "Claude Code Setup Commands:"
	@echo "  make setup       - Complete setup (permissions + verification)"
	@echo "  make permissions - Set executable permissions on hooks"

setup: permissions
	@echo "✅ Claude Code setup complete!"

permissions:
	@echo "Setting executable permissions on hooks..."
	@chmod +x hooks/*.py
	@chmod +x setup-hooks.sh
	@echo "✅ Permissions set!"
	@ls -la hooks/*.py | grep -E '\.py$$'