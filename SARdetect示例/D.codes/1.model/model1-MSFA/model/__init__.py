"""Model definitions for the SAR civil-vehicle two-stage baseline.

Submodules are intentionally not imported eagerly. This keeps detector config
inspection independent from classifier/runtime dependencies and makes operator
failures easier to isolate on Ascend.
"""
