"""Estimating tennis player ability under epsilon-contamination.

Pipeline stages are gated, not parallel (see README.md and docs/DATA_PIPELINE.md):
    fetch -> load -> blocks -> models -> evaluate
Keep each stage independently runnable; notebooks are never a dependency.
"""
