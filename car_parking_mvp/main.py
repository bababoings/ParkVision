"""
CETYS Parking Detection System - MVP
=====================================
Sistema de Detección de Disponibilidad de Estacionamiento en Tiempo Real.
Basado en el repositorio Chando0185/car_parking_detection_space_count,
adaptado para CETYS Universidad.

Pilares:
1. Carga dinámica de video (POST /upload)
2. Delimitación web de espacios (SpacePicker)
3. MobileNetV2 con Transfer Learning
4. API enriquecida con agrupación por zonas
"""

import os
import pickle
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

from preprocessing import preprocess_frame
import requests
from dotenv import load_dotenv

# Load env variables - use explicit path relative to this script
_env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(_env_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"[DEBUG] .env path: {_env_path}", flush=True)
print(f"[DEBUG] SUPABASE_URL loaded: {'YES' if SUPABASE_URL else 'NO'}", flush=True)
print(f"[DEBUG] SUPABASE_KEY loaded: {'YES' if SUPABASE_KEY else 'NO'}", flush=True)

# Initialize Supabase client
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client, Client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[INFO] Supabase client initialized.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to init Supabase: {e}", flush=True)
else:
    print("[WARNING] Supabase credentials not found. Sync disabled.", flush=True)
# =============================================================================
# App Configuration
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
POSITIONS_FILE = os.path.join(os.path.dirname(__file__), 'carposition.pkl')
LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), 'layouts')
ACTIVE_LAYOUT_FILE = os.path.join(os.path.dirname(__file__), 'active_layout.txt')
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'parking_mobilenetv2.h5')
MODEL_FALLBACK = os.path.join(os.path.dirname(__file__), 'model_final.h5')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LAYOUTS_DIR, exist_ok=True)

# =============================================================================
# Global Application State
# =============================================================================
app_state = {
    "video_path": None,
    "cap": None,
    "cap_lock": threading.Lock(),
    "active_layout": "default",
    "positions": [],          # List of {points: [[x1,y1],...,[x4,y4]], zone}
    "zones": {},              # {"Zona A": [indices...]}
    "model": None,
    "class_dictionary": {0: 'Disponible', 1: 'Ocupado'},
    "previous_states": [],    # For anti-flicker logic
    "confidence_threshold": 0.7,
    "model_input_size": (96, 96),  # MobileNetV2 input
}


def load_model_lazy():
    """Lazy-load the detection model on first use."""
    if app_state["model"] is not None:
        return app_state["model"]

    try:
        from tensorflow.keras.models import load_model as tf_load_model

        if os.path.exists(MODEL_FILE):
            print(f"[INFO] Loading model: {MODEL_FILE}")
            app_state["model"] = tf_load_model(MODEL_FILE)
            app_state["model_input_size"] = (96, 96)
        elif os.path.exists(MODEL_FALLBACK):
            print(f"[INFO] MobileNetV2 model not found, loading fallback: {MODEL_FALLBACK}")
            app_state["model"] = tf_load_model(MODEL_FALLBACK)
            app_state["model_input_size"] = (48, 48)
            app_state["class_dictionary"] = {0: 'Disponible', 1: 'Ocupado'}
        else:
            print("[WARNING] No model file found. Detection will not work.")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    return app_state["model"]


def load_positions(layout_name=None):
    """Load positions from Supabase or local fallback."""
    if layout_name is None:
        try:
            if os.path.exists(ACTIVE_LAYOUT_FILE):
                with open(ACTIVE_LAYOUT_FILE, "r") as f:
                    app_state["active_layout"] = f.read().strip()
            else:
                app_state["active_layout"] = "default"
        except Exception:
            app_state["active_layout"] = "default"
        layout_name = app_state["active_layout"]
    else:
        app_state["active_layout"] = layout_name
        try:
            with open(ACTIVE_LAYOUT_FILE, "w") as f:
                f.write(layout_name)
        except Exception:
            pass

    positions_data = []
    loaded_from_db = False

    if supabase:
        try:
            res = supabase.table('parking_layouts').select('positions').eq('name', layout_name).execute()
            if res.data and len(res.data) > 0:
                positions_data = res.data[0]['positions']
                loaded_from_db = True
                print(f"[INFO] Loaded layout '{layout_name}' from Supabase.")
            else:
                print(f"[INFO] Layout '{layout_name}' not found in Supabase.")
        except Exception as e:
            print(f"[ERROR] Failed to load layout from Supabase: {e}")

    if not loaded_from_db:
        # Fallback to local pkl
        pkl_path = os.path.join(LAYOUTS_DIR, f"{layout_name}.pkl")
        if not os.path.exists(pkl_path) and layout_name == "default":
            pkl_path = POSITIONS_FILE
            
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    positions_data = data
                print(f"[INFO] Loaded layout '{layout_name}' from local file.")
            except Exception as e:
                print(f"[ERROR] Failed to load local file: {e}")

    app_state["positions"] = positions_data if isinstance(positions_data, list) else []

    # Build zone index
    zones = {}
    for i, pos in enumerate(app_state["positions"]):
        zone_name = pos.get("zone", "Zona A")
        if zone_name not in zones:
            zones[zone_name] = []
        zones[zone_name].append(i)
    app_state["zones"] = zones

    # Initialize previous states
    app_state["previous_states"] = [None] * len(app_state["positions"])

    print(f"[INFO] Active config '{layout_name}' has {len(app_state['positions'])} positions in {len(app_state['zones'])} zones")


def init_video(video_source=None):
    """Initialize or reinitialize the video capture (supports webcam & files)."""
    with app_state["cap_lock"]:
        if app_state["cap"] is not None:
            app_state["cap"].release()

        # If it's a specific local path (for testing or uploaded videos)
        if video_source is not None and isinstance(video_source, str) and os.path.exists(video_source):
            app_state["video_path"] = video_source
            app_state["cap"] = cv2.VideoCapture(video_source)
            print(f"[INFO] Video loaded: {video_source}")
        else:
            # Switch to Webcam (C920 MVP Paradigm)
            camera_index = 0
            if video_source is not None and isinstance(video_source, int):
                camera_index = video_source
            elif video_source is not None and str(video_source).isdigit():
                camera_index = int(video_source)

            app_state["video_path"] = f"Webcam {camera_index}"
            # Use CAP_DSHOW for better webcam initialization on Windows
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            # C920 Settings: Full HD (1080p @ 30fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                print(f"[WARNING] Could not open webcam index {camera_index}. Trying default mp4...")
                default_video = os.path.join(os.path.dirname(__file__), 'car_test.mp4')
                if os.path.exists(default_video):
                    cap = cv2.VideoCapture(default_video)
                    app_state["video_path"] = default_video
                else:
                    cap = None
                    app_state["video_path"] = None

            app_state["cap"] = cap
            print(f"[INFO] Initialized capture source: {app_state['video_path']}")


# =============================================================================
# Detection Logic
# =============================================================================

def crop_with_volume(img, points, output_size=(96, 96)):
    """
    Extract image region using the axis-aligned bounding box of the polygon,
    extended upwards to capture the 3D volume (roof) of the vehicle.
    """
    h_img, w_img = img.shape[:2]
    pts = np.array(points, dtype=np.int32)

    if pts.shape != (4, 2):
        return None

    # Get axis-aligned bounding rectangle of the 4 ground points
    x, y, w, h = cv2.boundingRect(pts)

    # Expand bounding box upwards for car height (25% of height)
    # Expand slightly left/right/down (5%) to prevent tight clipping
    pad_up = int(h * 0.25)
    pad_side = int(w * 0.05)
    pad_down = int(h * 0.05)

    new_y = max(0, y - pad_up)
    new_x = max(0, x - pad_side)
    new_h = min(h_img - new_y, h + pad_up + pad_down)
    new_w = min(w_img - new_x, w + pad_side * 2)

    if new_w <= 0 or new_h <= 0:
        return None

    # Extract raw pixels — no perspective distortion!
    crop = img[new_y:new_y+new_h, new_x:new_x+new_w]

    # Resize to model input size
    resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    return resized


def check_parking_spaces(img):
    """
    Analyze parking spaces in the given frame.

    Uses dynamic volume-aware cropping (no destructive perspective warping)
    then the model to predict occupancy. Applies anti-flicker logic
    based on confidence threshold (VPD-01 Alternative).

    Returns:
        tuple: (annotated_img, available_count, occupied_count, zone_stats)
    """
    model = load_model_lazy()
    positions = app_state["positions"]

    if model is None or len(positions) == 0:
        return img, 0, 0, {}

    input_size = app_state["model_input_size"]
    img_crops = []

    for pos in positions:
        points = pos.get("points", [])
        if len(points) != 4:
            img_crops.append(np.zeros((*input_size, 3)))
            continue

        try:
            crop = crop_with_volume(img, points, input_size)
            if crop is not None:
                normalized = crop / 255.0
                img_crops.append(normalized)
            else:
                img_crops.append(np.zeros((*input_size, 3)))
        except Exception:
            img_crops.append(np.zeros((*input_size, 3)))

    img_crops = np.array(img_crops)
    predictions = model.predict(img_crops, verbose=0)

    available = 0
    occupied = 0
    zone_stats = {}

    for i, pos in enumerate(positions):
        points = pos.get("points", [])
        if len(points) != 4:
            continue

        zone = pos.get("zone", "Zona A")
        pts_np = np.array(points, dtype=np.int32)

        confidence = float(np.max(predictions[i]))
        predicted_class = int(np.argmax(predictions[i]))

        # Anti-flicker: if confidence is low, keep previous state
        if confidence < app_state["confidence_threshold"]:
            if app_state["previous_states"][i] is not None:
                predicted_class = app_state["previous_states"][i]
        else:
            app_state["previous_states"][i] = predicted_class

        label = app_state["class_dictionary"].get(predicted_class, "Unknown")
        is_available = (predicted_class == 0)

        if is_available:
            color = (0, 230, 118)   # Green
            thickness = 3
            text_color = (0, 0, 0)
            available += 1
        else:
            color = (82, 82, 255)   # Red (BGR)
            thickness = 2
            text_color = (255, 255, 255)
            occupied += 1

        # Draw quadrilateral
        cv2.polylines(img, [pts_np], isClosed=True, color=color,
                      thickness=thickness)
        # Semi-transparent fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_np], color)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # Draw label at centroid commented out to prevent stacking/visual clutter
        # cx = int(np.mean(pts_np[:, 0]))
        # cy = int(np.mean(pts_np[:, 1]))
        # display_text = f"{label} ({confidence:.0%})"
        # font_scale = 0.45
        # text_thickness = 1
        # text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX,
        #                             font_scale, text_thickness)[0]
        # text_x = cx - text_size[0] // 2
        # text_y = cy + text_size[1] // 2
        # cv2.rectangle(img, (text_x - 3, text_y - text_size[1] - 5),
        #               (text_x + text_size[0] + 3, text_y + 2), color, -1)
        # cv2.putText(img, display_text, (text_x, text_y - 3),
        #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
        #             text_thickness)

        # Zone aggregation (ZAM-01)
        if zone not in zone_stats:
            zone_stats[zone] = {"total": 0, "available": 0, "occupied": 0}
        zone_stats[zone]["total"] += 1
        zone_stats[zone]["available" if is_available else "occupied"] += 1

    # Calculate availability confidence per zone (ratio of available spaces to total)
    for zone, stats in zone_stats.items():
        if stats["total"] > 0:
            av_confidence = stats["available"] / stats["total"]
            stats["confidence"] = round(av_confidence, 2)
        else:
            stats["confidence"] = 0.0

    return img, available, occupied, zone_stats


def generate_frames():
    """
    Generator that yields MJPEG frames for the video feed.
    Implements periodic frame capture (CI-01) and quality validation (CI-02).
    """
    while True:
        with app_state["cap_lock"]:
            cap = app_state["cap"]
            if cap is None or not cap.isOpened():
                break

            success, img = cap.read()

            # Loop video when finished (only if playing a file, not webcam)
            if not success:
                is_file = str(app_state.get("video_path", "")).endswith(('.mp4', '.avi', '.mkv', '.mov'))
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, img = cap.read()
                
                if not success:
                    break

        # Preprocessing pipeline (CI-02)
        processed, is_valid, issues = preprocess_frame(img)
        if processed is None:
            continue

        if not is_valid:
            # Even if not ideal quality, still process but could log warnings
            pass

        # Run detection
        annotated, _, _, _ = check_parking_spaces(processed)

        ret, buffer = cv2.imencode('.jpg', annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Frame interval control (CI-01)
        time.sleep(0.03)  # ~30 FPS cap


# =============================================================================
# Routes
# =============================================================================
@app.route('/')
def index():
    """Dashboard principal."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream de video MJPEG."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/space_count')
def space_count():
    """
    API enriquecida de conteo de espacios.
    Devuelve total, disponibles, ocupados, timestamp y desglose por zonas.
    """
    with app_state["cap_lock"]:
        cap = app_state["cap"]
        if cap is None or not cap.isOpened():
            return jsonify(
                total=len(app_state["positions"]),
                available=0,
                occupied=0,
                timestamp=datetime.now().isoformat(),
                zones={}
            )

        success, img = cap.read()
        if not success:
            is_file = str(app_state.get("video_path", "")).endswith(('.mp4', '.avi', '.mkv', '.mov'))
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, img = cap.read()

    if success:
        processed, _, _ = preprocess_frame(img)
        if processed is not None:
            _, avail, occ, zone_stats = check_parking_spaces(processed)
            return jsonify(
                total=avail + occ,
                available=avail,
                occupied=occ,
                timestamp=datetime.now().isoformat(),
                zones=zone_stats
            )

    return jsonify(
        total=0,
        available=0,
        occupied=0,
        timestamp=datetime.now().isoformat(),
        zones={}
    )


@app.route('/upload', methods=['GET'])
def upload_page():
    """Página de carga de video."""
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Endpoint para subir un archivo de video."""
    if 'video' not in request.files:
        return jsonify(error="No se envió ningún archivo"), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify(error="Archivo sin nombre"), 400

    allowed_ext = {'.mp4', '.avi', '.mov', '.mkv'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify(error=f"Formato no soportado: {ext}"), 400

    # Save file
    filename = f"video{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Reinitialize video capture
    init_video(filepath)

    return jsonify(
        success=True,
        message=f"Video '{file.filename}' cargado correctamente.",
        filename=filename
    )


@app.route('/space_picker')
def space_picker():
    """Página del SpacePicker interactivo."""
    return render_template('space_picker.html')


@app.route('/first_frame')
def first_frame():
    """Devuelve el primer frame del video actual como JPEG."""
    with app_state["cap_lock"]:
        cap = app_state["cap"]
        if cap is None or not cap.isOpened():
            # Return a placeholder image
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No video loaded",
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (100, 100, 100), 2)
            ret, buf = cv2.imencode('.jpg', placeholder)
            return Response(buf.tobytes(), mimetype='image/jpeg')

        # Save current position, go to frame 0, read, restore
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    if not success:
        return jsonify(error="Could not read frame"), 500

    frame = cv2.resize(frame, (1280, 720))
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/save_positions', methods=['POST'])
def save_positions():
    """Guarda las posiciones definidas desde el SpacePicker en Supabase (o local)."""
    data = request.get_json()
    if not data or 'positions' not in data:
        return jsonify(error="No se recibieron posiciones"), 400

    positions = data['positions']
    layout_name = data.get('layout_name', app_state.get('active_layout', 'default'))

    if not layout_name.strip():
        layout_name = "default"

    # Validate: each position must have 'points' (4 items of [x,y]) and 'zone'
    for i, p in enumerate(positions):
        if 'points' not in p or 'zone' not in p:
            return jsonify(error=f"Cajón {i+1}: formato inválido (falta 'points' o 'zone')"), 400
        if not isinstance(p['points'], list) or len(p['points']) != 4:
            return jsonify(error=f"Cajón {i+1}: se requieren exactamente 4 puntos"), 400
        for j, pt in enumerate(p['points']):
            if not isinstance(pt, list) or len(pt) != 2:
                return jsonify(error=f"Cajón {i+1}, punto {j+1}: formato inválido"), 400

    # Save to Supabase
    saved_to_db = False
    if supabase:
        try:
            supabase.table('parking_layouts').upsert({
                "name": layout_name,
                "positions": positions
            }, on_conflict="name").execute()
            saved_to_db = True
        except Exception as e:
            print(f"[ERROR] Failed to save layout '{layout_name}' to Supabase: {e}")

    # Fallback/Mirror to local backup
    try:
        if layout_name == "default":
            pkl_path = POSITIONS_FILE
        else:
            pkl_path = os.path.join(LAYOUTS_DIR, f"{layout_name}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(positions, f)
    except Exception as e:
        if not saved_to_db:
            return jsonify(error=f"Error al guardar base local: {str(e)}"), 500

    # Update app state if saving to active layout
    if layout_name == app_state.get('active_layout', 'default'):
        load_positions(layout_name)

    return jsonify(success=True, count=len(positions))


@app.route('/load_positions')
def load_positions_route():
    """Carga las posiciones existentes como JSON."""
    return jsonify(positions=app_state["positions"])


@app.route('/api/layouts', methods=['GET'])
def get_layouts():
    """Retorna la lista de layouts disponibles."""
    layouts = []
    
    if supabase:
        try:
            res = supabase.table('parking_layouts').select('name').execute()
            if res.data:
                layouts.extend([item['name'] for item in res.data])
        except Exception as e:
            print(f"[ERROR] Failed to fetch layouts from Supabase: {e}")
            
    local_files = []
    if os.path.exists(POSITIONS_FILE):
        local_files.append('default')
        
    if os.path.exists(LAYOUTS_DIR):
        for f in os.listdir(LAYOUTS_DIR):
            if f.endswith('.pkl'):
                name = f[:-4]
                if name != 'default' or 'default' not in local_files:
                    local_files.append(name)
                
    layouts.extend([l for l in local_files if l not in layouts])
    if not layouts:
        layouts = ['default']
        
    return jsonify({
        "layouts": list(set(layouts)),
        "active_layout": app_state.get("active_layout", "default")
    })

@app.route('/api/layouts/active', methods=['POST'])
def set_active_layout():
    """Cambia el layout activo."""
    data = request.get_json()
    new_layout = data.get('layout')
    if not new_layout:
        return jsonify(error="Nombre de layout vacío"), 400
        
    load_positions(new_layout)
    return jsonify(success=True, active_layout=app_state["active_layout"])

@app.route('/api/layouts/<layout_name>', methods=['GET'])
def get_layout(layout_name):
    """Obtiene las posiciones de un layout específico sin volverlo activo."""
    positions_data = []
    loaded = False
    
    if supabase:
        try:
            res = supabase.table('parking_layouts').select('positions').eq('name', layout_name).execute()
            if res.data and len(res.data) > 0:
                positions_data = res.data[0]['positions']
                loaded = True
        except Exception as e:
            pass
            
    if not loaded:
        pkl_path = os.path.join(LAYOUTS_DIR, f"{layout_name}.pkl")
        if not os.path.exists(pkl_path) and layout_name == "default":
            pkl_path = POSITIONS_FILE
            
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    positions_data = data
            except:
                pass
                
    return jsonify(positions=positions_data)


# =============================================================================
# Startup
# =============================================================================

def sync_supabase_worker():
    """Background thread to sync data to Supabase periodically."""
    print("[INFO] Supabase sync background worker started.")
    time.sleep(5)  # Wait for startup
    while True:
        try:
            if not supabase:
                time.sleep(30)
                continue

            with app_state["cap_lock"]:
                cap = app_state["cap"]
                if cap is None or not cap.isOpened():
                    time.sleep(5)
                    continue

                success, img = cap.read()
                if not success:
                    is_file = str(app_state.get("video_path", "")).endswith(('.mp4', '.avi', '.mkv', '.mov'))
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        success, img = cap.read()
            
            if success:
                processed, _, _ = preprocess_frame(img)
                if processed is not None:
                    _, avail, occ, zone_stats = check_parking_spaces(processed) # fetch de informacion para actualizar/insertar en la base de datos
                    
                    records = []
                    current_time_iso = datetime.now().isoformat()
                    
                    for zone_id, stats in zone_stats.items():
                        records.append({
                            "zone_id": zone_id,
                            "available_spaces": stats["available"],
                            "occupied_spaces": stats["occupied"],
                            "confidence": stats.get("confidence", 1.0),
                            "updated_at": current_time_iso
                        })
                    
                    if records:
                        response = supabase.table('occupancy').upsert(records).execute()
                        print(f"[{current_time_iso}] Supabase Sync: {len(records)} zones sent.")

        except Exception as e:
            print(f"[ERROR] Sync worker: {e}")

        time.sleep(15)  # Sincroniza cada 15 segundos

# Load positions and initialize video at startup
load_positions()
init_video()

# Start sync thread
threading.Thread(target=sync_supabase_worker, daemon=True).start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
