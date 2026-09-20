"""Estimating tennis player ability under epsilon-contamination.

Pipeline stages are gated, not parallel (see CLAUDE.md / tennis-contamination.md):
    fetch -> load -> blocks -> models -> evaluate
Keep each stage independently runnable; notebooks are never a dependency.
"""
