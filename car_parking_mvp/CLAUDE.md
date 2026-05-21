# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CETYS Parking Detection MVP. A Flask app that classifies parking spaces as "Disponible" (empty) or "Ocupado" (occupied) using a MobileNetV2 transfer-learning model on cropped quadrilateral ROIs from a video/webcam feed. Detailed walkthrough in `WALKTHROUGH.md`; cropping-algorithm rationale in `ALGORITMO_VOLUMETRICO.md`.

## Common Commands

All commands assume the project venv is activated (`source venv/bin/activate` on macOS/Linux, `.\venv\Scripts\Activate.ps1` on Windows). Always install via the venv's explicit Python on Windows (`.\venv\Scripts\python.exe -m pip install ...`) — installing globally has bitten this project (see `WALKTHROUGH.md` §6).

```bash
pip install -r requirements.txt    # Install deps
python main.py                     # Run Flask app on :5000 (loads positions + opens webcam/video)
python train_model.py              # Train parking_mobilenetv2.h5 from train_data/{train,test}/{empty,occupied}/
python evaluate_model.py           # Score model -> metrics_output/{confusion_matrix,roc_curve}.png
python process_pklot_coco.py       # Rebuild train_data/ from a PKLot COCO dump (paths hardcoded for Windows)
python datacollection.py           # Standalone OpenCV space-picker over car1.png (mostly superseded by /space_picker)
```

There is no test suite, linter, or build step.

## Architecture

### Runtime flow (`main.py`)
1. At import time: `load_positions()` reads the active layout, `init_video()` opens the capture, and a daemon `sync_supabase_worker` thread is started. **This means importing `main` has side effects** (opens the webcam, hits Supabase) — be careful with import order in any new tooling.
2. `/video_feed` streams MJPEG by running `generate_frames()` → `preprocess_frame()` → `check_parking_spaces()` per frame.
3. `/space_count` reads a single frame on demand and returns the same `(available, occupied, zone_stats)` payload as JSON. The Supabase sync worker calls the same path every 15s and upserts into the `occupancy` table.
4. The shared capture handle lives in `app_state["cap"]` guarded by `app_state["cap_lock"]` — every reader (the stream, the API endpoint, the sync worker, `/first_frame`) must hold the lock while calling `cap.read()`.

### Detection pipeline
- **`preprocessing.preprocess_frame`** validates resolution + brightness and resizes to 1280×720. Returns `(frame, is_valid, issues)`; an invalid frame is still passed through (current code only logs).
- **`crop_with_volume(img, points, output_size)`** is the cropping routine the model was trained for. It takes the axis-aligned bounding box of the 4 ground points, then pads it (25% up, 5% sides/down) to capture the vehicle's 3D volume, then extracts raw pixels and resizes. **Do not replace this with `cv2.warpPerspective` / homography** — the model was specifically retrained on PKLot to handle the perspective-preserving crop, and the prior DLT approach is what was deliberately replaced (see `ALGORITMO_VOLUMETRICO.md`).
- **Anti-flicker**: predictions with confidence < `app_state["confidence_threshold"]` (0.7) reuse the previous class from `app_state["previous_states"]`. The state array is reset on every `load_positions()` call.
- **Class dictionary**: `0 → Disponible`, `1 → Ocupado`. The legacy `model_final.h5` fallback uses 48×48 input; `parking_mobilenetv2.h5` uses 96×96. `app_state["model_input_size"]` is set based on which file loads.

### Layouts and persistence
- A *layout* is a list of `{"points": [[x,y]*4], "zone": "Zona X"}` dicts. The `points` must be exactly 4 — `/save_positions` validates this and `check_parking_spaces` silently skips anything else.
- The active layout name is persisted in `active_layout.txt`. Each layout has two storage tiers:
  - **Supabase** `parking_layouts` table (`name`, `positions` JSON) — primary if `SUPABASE_URL`/`SUPABASE_KEY` are in `.env`.
  - **Local pickle fallback**: the `default` layout lives at `carposition.pkl` (top level); all others at `layouts/<name>.pkl`. Saves always mirror locally even when Supabase succeeds.
- `/api/layouts/active` swaps the active layout at runtime — it calls `load_positions(name)`, which also resets `previous_states`.

### Video source
- `init_video()` defaults to **webcam index 0** opened with `cv2.CAP_DSHOW` and forced to 1920×1080@30fps (this is the Logitech C920 MVP target). `CAP_DSHOW` is Windows-only; on macOS/Linux the flag is ignored but a working webcam still opens. If the webcam fails, it falls back to `car_test.mp4` at the project root.
- `/upload` (POST) accepts mp4/avi/mov/mkv, saves to `uploads/video<ext>` (always the same filename — uploads overwrite), then calls `init_video(filepath)` to switch sources live.
- When the source is a file (extension in `.mp4/.avi/.mkv/.mov`), the stream loops by resetting `CAP_PROP_POS_FRAMES`. Webcams don't loop.

### Templates and static
- `templates/index.html` — dashboard consuming `/video_feed` and polling `/space_count`.
- `templates/space_picker.html` — pulls `/first_frame` and lets the user draw 4-point quads in JS, then POSTs to `/save_positions` with a `layout_name`.
- `templates/upload.html` — file picker → POST `/upload`.

### Supabase integration
- Credentials in `.env` at the project root (`SUPABASE_URL`, `SUPABASE_KEY`); both must be set or the client stays `None` and sync is disabled silently.
- The `occupancy` table is upserted on `zone_id` with `(available_spaces, occupied_spaces, confidence, updated_at)`. **`confidence` here is the availability ratio** (`available / total` per zone), not model confidence — see `WALKTHROUGH.md` §6 before changing the meaning.
- The `parking_spaces` table is upserted on `space_id` (one row per individual spot, FK `zone_id` → `zones`). Geometry + zone link are written by `_sync_spaces_to_db` from `/save_positions` (replace) and `load_positions` (upsert); the sync worker then patches only `is_occupied`, `confidence`, `updated_at` every tick. `confidence` here is **per-space model confidence** — different meaning than the `occupancy.confidence` column. Space IDs are deterministic: `f"{layout_name}-{index+1:03d}"`. Reordering or deleting a space shifts every later ID — accept this for the mobile MVP. Mobile app consumes this table directly.
- `parking_layouts` upserts on `name`. If using the anon key, RLS policies must allow insert/update.

## Gotchas
- `parking_mobilenetv2.h5`, `*.pkl`, `train_data/`, `uploads/`, `venv/`, and `.env` are all gitignored — they don't ship with a fresh clone. `WALKTHROUGH.md` §2 has the full reconstruction steps.
- `process_pklot_coco.py` has hardcoded Windows paths (`D:\Pklot_dataset`, `train_data\train`) — adjust before running on macOS/Linux.
- `datacollection.py` is the standalone (non-Flask) space picker for `car1.png`. It uses the same `carposition.pkl` format as the web flow.
- Flask runs with `debug=True, use_reloader=False`. The reloader is deliberately off because the import-time side effects (webcam open, sync thread) would double up.
