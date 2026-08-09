# RoomScan Energy Audit

Guidance specific to the RoomScan energy-audit subsystem: `aria_capture.py` (dual-backend VRS + live sensor capture with ring-buffered fan-out) + `capture_healthcheck.py` (verification harness), consumed by the energy pipeline: `energy_detector.py` (YOLOv8 appliance detection) -> `energy_estimator.py` (catalog kWh/cost math) -> `roomscan.py` (orchestrator CLI) -> `energy_report.py` (self-contained HTML report). See `CLAUDE.md` for shared environment setup, conventions, and the air-writing pipeline (`AIRWRITING.md`).

All commands below are run from the repo root (each script adds the repo root back onto `sys.path` itself, so this works whether or not you `cd` into `roomscan/` first).

## Commands

### Aria capture healthcheck

```bash
python3 roomscan/capture_healthcheck.py --vrs /path/to/recording.vrs [--duration 30] [--out healthcheck_out]
python3 roomscan/capture_healthcheck.py --live [--start-streaming --device-ip <ip> --interface usb|wifi --profile profile18]
```
Exit code 0 only if every expected stream (camera-rgb, camera-slam-left/right, camera-et-left/right, imu-right, imu-left, mag0, baro0) is alive and timestamp-monotonic (plus, in live mode, RGB/IMU skew < 100 ms). Writes one upright sample JPEG per camera to `healthcheck_out/` for visual orientation/eye-split verification.

### RoomScan energy audit

```bash
python3 roomscan/roomscan.py --vrs /path/to/walkthrough.vrs --room-name "Living room" [--out roomscan_out]
python3 roomscan/roomscan.py --live [--start-streaming --device-ip <ip> --interface usb|wifi] [--duration 60]
python3 roomscan/energy_report.py --json roomscan_out/roomscan_report.json   # regenerate HTML only
python3 roomscan/energy_detector.py --vrs /path/to/recording.vrs             # detection-only debug scan
```
Requires `ultralytics` (auto-downloads `yolov8n.pt` on first run; gitignored via `*.pt`). Outputs `roomscan_report.json`, per-instance crops, and a self-contained `roomscan_report.html` (base64-inlined crops — openable anywhere with no server). Exit 0 if appliances were found, 2 if none, per `roomscan.py:main()`.

Optional: `pip install google-genai` + `export GEMINI_API_KEY=...` (never commit the key) upgrades the report's `recommendations` field to Gemini-vision-generated, photo-grounded suggestions; without a key set, or if the call fails for any reason, it silently falls back to the rule-based engine — see `energy_gemini.py`.

Live dashboard: `python3 roomscan/roomscan_dashboard.py [--device-ip <ip> --start-streaming --interface usb --profile profile18] [--out roomscan_out]` drives `roomscan_live.py:LiveScanController` directly (no WebSocket/bridge involved, unlike the air-writing dashboard). Every `GEMINI_LIVE_PASS_INTERVAL_SECONDS` (5s) it runs a combined verify+discover Gemini pass; the dashboard's right-hand "Gemini's Last Look" panel shows the exact frame that pass analyzed plus a caption of anything newly identified in it (`LiveScanController.snapshot()`'s `gemini_last_pass_frame` / `gemini_last_pass_new_items` fields, consumed by `roomscan_dashboard.py:_update_gemini_snapshot()`).

## Architecture

### Aria capture layer (`aria_capture.py`)

Single `AriaCapture` class, two backends behind one callback interface (`source="vrs"` or `source="live"`):
- **VRS backend**: `projectaria_tools.core.data_provider.create_vrs_data_provider`, streams resolved by label (never hardcoded stream IDs, via `CAPTURE_VRS_LABEL_ALIASES` in config), played back with `deliver_queued_sensor_data()` for device-time-ordered interleaving across all modalities (not per-stream index loops).
- **Live backend**: `aria.sdk.StreamingClient` + observer, imported lazily so VRS-only environments never need the Client SDK installed.
- **Gen 1 eye tracking is one physical stream** (`camera-et`) with both eyes side by side in one image; both backends split it at the horizontal midpoint into `camera-et-left` / `camera-et-right` sharing one timestamp.
- **Orientation contract**: images are delivered RAW (un-rotated, native sensor frame — required for calibration/undistortion); `rotate_upright()` (`np.rot90(frame, -1)`) is a separate helper for display-only consumers. Never assume a callback frame is display-oriented.
- **Timestamps**: every sample carries device-time nanoseconds (`capture_timestamp_ns`); wall-clock arrival time is never used for cross-sensor alignment.
- **Fan-out**: producer threads only write buffers (single-slot latest-value for images, `deque(maxlen=2000)` for IMU/mag/baro) — one dispatcher thread invokes subscriber callbacks, so a slow subscriber can never block capture.
- `get_calibration(label)` returns `CameraCalibration` from `provider.get_device_calibration()` in VRS mode; always `None` in live mode (the Client SDK streaming path doesn't deliver device calibration in this build).

`capture_healthcheck.py` is the verification harness for this layer.

### Energy audit pipeline (`roomscan.py` and friends)

`roomscan.py` orchestrates: `AriaCapture` (either backend) -> `energy_detector.scan_capture_rgb()` subscribes to camera-rgb, samples frames at ~2 Hz **device time**, rotates RAW frames upright before YOLO -> `ApplianceScanAggregator` counts instances with the **max-simultaneous rule** (per class, count = most detections seen in any single frame; pan-away/pan-back never double-counts), disambiguating genuinely distinct never-simultaneous instances of the same class via appearance-based re-identification (color-histogram similarity, `ENERGY_REID_SIMILARITY_THRESHOLD` in config), and keeps the best-confidence crop per instance slot -> `energy_estimator.estimate_room()` maps counts through `ENERGY_CATALOG` priors -> `energy_report.render_html()` writes the self-contained page. Two subtleties: (1) in VRS mode `scan_capture_rgb(pace_playback=True)` subscribes a no-op imu-right consumer to engage the capture layer's backpressure — without it, faster-than-realtime playback plus the drop-stale image slot starves slow YOLO inference down to a few frames per file; live mode must keep `pace_playback=False` (drop-stale is correct there). (2) `ApplianceScanAggregator` and `energy_estimator` are deliberately torch-free (ultralytics is lazily imported inside `EnergyDetector`) so `tests/test_energy.py` runs without YOLO.

`roomscan.py:build_report()`'s `recommendations` field comes from `energy_gemini.get_recommendations()`, which sends the scan's best-confidence crops (biggest energy users first, capped at `GEMINI_MAX_CROPS`) to Gemini vision when `GEMINI_API_KEY` is set, and otherwise — or on any Gemini failure — falls back to `energy_recommendations.generate_recommendations()`'s pure rule engine. This is deliberately the *only* call site that can hit the network: the live dashboard's per-tick recommendations panel and its instant Stop-Scan summary dialog (`roomscan_dashboard.py`) call `generate_recommendations()` directly and always stay rule-based, since a network call on a ~1s UI tick would be reckless. `energy_gemini.py` mirrors `energy_detector.py`'s lazy-import trick (the `google-genai` SDK is only imported inside `_generate_content()`), so `tests/test_energy_gemini.py` — like `tests/test_energy.py` — never needs the optional dependency installed.

`roomscan_live.py:LiveScanController` is the live-dashboard backend: runs continuously, exposes incremental scan state via `snapshot()` at any point (not just a final report), and reuses `roomscan.py`'s `finalize_scan()` verbatim at session end so live and batch (`--vrs`/`--live`) runs produce identical artifacts. Its background `_gemini_pass_loop()` thread runs one combined verify+discover Gemini pass every `GEMINI_LIVE_PASS_INTERVAL_SECONDS`, skipping a tick outright (never queuing) if the previous call is still in flight; it tracks both the cumulative all-session `_gemini_discovered` list and the single most-recent pass's frame/new-item-names (`_gemini_last_pass_frame` / `_gemini_last_pass_new_items`) for the dashboard's per-pass "here's what Gemini just looked at" display.
