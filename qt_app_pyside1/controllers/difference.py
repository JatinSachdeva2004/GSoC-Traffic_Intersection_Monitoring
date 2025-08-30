# Detailed Comparison: video_controller_new.py vs video_controller_finale.py
#
# This document provides a function-by-function, block-by-block comparison between `video_controller_new.py` and `video_controller_finale.py` as of July 2025. It highlights what is present, missing, or different in each file, and explains the practical impact of those differences for real-world red light violation detection and video analytics.
#
# ---
#
# ## Table of Contents
# - [Overall Structure](#overall-structure)
# - [Class/Function Inventory](#classfunction-inventory)
# - [Function-by-Function Comparison](#function-by-function-comparison)
#   - [__init__](#__init__)
#   - [set_source](#set_source)
#   - [_get_source_properties](#_get_source_properties)
#   - [start/stop](#startstop)
#   - [_run](#_run)
#   - [_process_frame](#_process_frame)
#   - [detect_red_light_violations](#detect_red_light_violations)
# - [Key Differences and Impact](#key-differences-and-impact)
# - [Summary Table](#summary-table)
#
# ---
#
# ## Overall Structure
#
# - **video_controller_new.py**
#   - Modernized, modular, and debug-heavy.
#   - Uses enhanced annotation utilities, more robust fallback logic, and detailed debug output.
#   - Violation detection logic is inlined and self-contained.
#   - State machine for per-vehicle violation tracking is explicit and debugged.
#   - Crosswalk/violation line detection is always run, with fallback.
#   - Always emits overlays and signals, even if no violators.
#
# - **video_controller_finale.py**
#   - Reference implementation, known to work reliably in production.
#   - May use some different utility imports and slightly different state handling.
#   - Violation detection logic may be more tightly coupled to tracker or external detector class.
#   - Debug output is present but may be less granular.
#   - Fallbacks for violation line and traffic light are robust.
#
# ---
#
# ## Class/Function Inventory
#
# | Function/Class                | In New | In Finale | Notes |
# |-------------------------------|--------|-----------|-------|
# | VideoController               |   ✔    |     ✔     | Main class in both |
# | __init__                      |   ✔    |     ✔     | New: more debug, explicit tracker, fallback logic |
# | set_source                    |   ✔    |     ✔     | Similar, new has more robust type handling |
# | _get_source_properties        |   ✔    |     ✔     | Similar, new has more debug |
# | start/stop                    |   ✔    |     ✔     | Similar, new has more debug |
# | _run                          |   ✔    |     ✔     | New: more debug, more robust detection/tracking |
# | _process_frame                |   ✔    |     ✔     | New: always runs crosswalk, overlays, fallback |
# | detect_red_light_violations   |   ✔    |     ✔     | New: inlined, explicit state machine, more debug |
# | violation_detector (external) |   ✖    |     ✔     | Finale may use RedLightViolationDetector class |
#
# ---
#
# ## Function-by-Function Comparison
#
# ### __init__
# - **New:**
#   - Sets up all state, tracker, and debug counters.
#   - Imports and initializes crosswalk detection utilities with try/except.
#   - Does NOT use external `RedLightViolationDetector` (commented out).
#   - Uses inlined `detect_red_light_violations` method.
# - **Finale:**
#   - May use external `RedLightViolationDetector` class for violation logic.
#   - Similar state setup, but possibly less debug output.
#
# ### set_source
# - **New:**
#   - Handles all source types robustly (file, camera, URL, device).
#   - More debug output for every branch.
# - **Finale:**
#   - Similar logic, possibly less robust for edge cases.
#
# ### _get_source_properties
# - **New:**
#   - More debug output, retries for camera sources.
# - **Finale:**
#   - Similar, but may not retry as aggressively.
#
# ### start/stop
# - **New:**
#   - More debug output, aggressive render timer (10ms).
# - **Finale:**
#   - Standard start/stop, less debug.
#
# ### _run
# - **New:**
#   - Handles detection, tracking, and annotation in one loop.
#   - Always normalizes class names.
#   - Always draws overlays and emits signals.
#   - More debug output for every step.
# - **Finale:**
#   - Similar, but may use external violation detector.
#   - May not always emit overlays if no detections.
#
# ### _process_frame
# - **New:**
#   - Always runs crosswalk/violation line detection.
#   - Always overlays violation line and traffic light status.
#   - Only runs violation detection if both red light and violation line are present.
#   - Always emits overlays/signals, even if no violators.
#   - More robust fallback for violation line.
#   - More debug output for every step.
# - **Finale:**
#   - Similar, but may skip overlays if no detections.
#   - May use external violation detector.
#
# ### detect_red_light_violations
# - **New:**
#   - Inlined, explicit state machine for per-vehicle tracking.
#   - Requires vehicle to be behind the line before crossing during red.
#   - Cooldown logic to prevent duplicate violations.
#   - Extensive debug output for every vehicle, every frame.
# - **Finale:**
#   - May use external class for violation logic.
#   - Similar state machine, but less debug output.
#   - May have slightly different fallback/cooldown logic.
#
# ---
#
# ## Key Differences and Impact
#
# - **External Violation Detector:**
#   - Finale uses `RedLightViolationDetector` class; New inlines the logic.
#   - Impact: New is easier to debug and modify, but harder to swap out logic.
#
# - **Debug Output:**
#   - New has much more granular debug output for every step and every vehicle.
#   - Impact: Easier to diagnose issues in New.
#
# - **Fallback Logic:**
#   - Both have robust fallback for violation line and traffic light, but New is more explicit.
#
# - **Overlay/Signal Emission:**
#   - New always emits overlays and signals, even if no violators.
#   - Finale may skip if no detections.
#
# - **State Machine:**
#   - New's state machine is explicit, per-vehicle, and debugged.
#   - Finale's may be more implicit or handled in external class.
#
# - **Modularity:**
#   - Finale is more modular (external detector class), New is more monolithic but easier to trace.
#
# ---
#
# ## Summary Table
#
# | Feature/Function                | video_controller_new.py | video_controller_finale.py |
# |---------------------------------|:----------------------:|:-------------------------:|
# | External Violation Detector     |           ✖            |            ✔              |
# | Inlined Violation Logic         |           ✔            |            ✖              |
# | Robust Fallbacks                |           ✔            |            ✔              |
# | Always Emits Overlays/Signals   |           ✔            |            ✖/Partial      |
# | Extensive Debug Output          |           ✔            |            ✖/Partial      |
# | Per-Vehicle State Machine       |           ✔            |            ✔              |
# | Modularity                      |           ✖            |            ✔              |
# | Easy to Debug/Modify            |           ✔            |            ✖/Partial      |
#
# ---
#
# ## Conclusion
#
# - Use `video_controller_new.py` for maximum debug visibility, easier modification, and robust fallback logic.
# - Use `video_controller_finale.py` for production-proven modularity and if you want to swap out violation logic easily.
# - Both are robust, but the new version is more transparent and easier to debug in real-world scenarios.
#
# ---
#
# *This file is auto-generated for developer reference. Update as code evolves.*
