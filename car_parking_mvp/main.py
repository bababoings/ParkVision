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

# =============================================================================
# App Configuration
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
POSITIONS_FILE = os.path.join(os.path.dirname(__file__), 'carposition.pkl')
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'parking_mobilenetv2.h5')
MODEL_FALLBACK = os.path.join(os.path.dirname(__file__), 'model_final.h5')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# Global Application State
# =============================================================================
app_state = {
    "video_path": None,
    "cap": None,
    "cap_lock": threading.Lock(),
    "positions": [],          # List of {x, y, w, h, zone}
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


def load_positions():
    """Load positions from pickle file, supporting old and new formats."""
    if not os.path.exists(POSITIONS_FILE):
        app_state["positions"] = []
        app_state["zones"] = {}
        return

    try:
        with open(POSITIONS_FILE, 'rb') as f:
            data = pickle.load(f)

        # Support new format: list of dicts with {x, y, w, h, zone}
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                app_state["positions"] = data
            else:
                # Legacy format: list of tuples (x, y)
                # Convert to new format with default w=130, h=65
                app_state["positions"] = [
                    {"x": pos[0], "y": pos[1], "w": 130, "h": 65, "zone": "Zona A"}
                    for pos in data
                ]
        else:
            app_state["positions"] = []

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

        print(f"[INFO] Loaded {len(app_state['positions'])} positions in "
              f"{len(app_state['zones'])} zones")
    except Exception as e:
        print(f"[ERROR] Failed to load positions: {e}")
        app_state["positions"] = []
        app_state["zones"] = {}


def init_video(video_path=None):
    """Initialize or reinitialize the video capture."""
    with app_state["cap_lock"]:
        if app_state["cap"] is not None:
            app_state["cap"].release()

        if video_path and os.path.exists(video_path):
            app_state["video_path"] = video_path
            app_state["cap"] = cv2.VideoCapture(video_path)
            print(f"[INFO] Video loaded: {video_path}")
        else:
            # Try default video
            default_video = os.path.join(os.path.dirname(__file__), 'car_test.mp4')
            if os.path.exists(default_video):
                app_state["video_path"] = default_video
                app_state["cap"] = cv2.VideoCapture(default_video)
                print(f"[INFO] Default video loaded: {default_video}")
            else:
                app_state["video_path"] = None
                app_state["cap"] = None
                print("[WARNING] No video file available.")


# =============================================================================
# Detection Logic
# =============================================================================
def check_parking_spaces(img):
    """
    Analyze parking spaces in the given frame.

    Uses the model to predict occupancy and applies anti-flicker logic
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
        x, y, w, h = pos["x"], pos["y"], pos["w"], pos["h"]
        # Clamp coordinates to frame boundaries
        y1 = max(0, y)
        y2 = min(img.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(img.shape[1], x + w)

        if y2 <= y1 or x2 <= x1:
            img_crops.append(np.zeros((*input_size, 3)))
            continue

        crop = img[y1:y2, x1:x2]
        resized = cv2.resize(crop, input_size)
        normalized = resized / 255.0
        img_crops.append(normalized)

    img_crops = np.array(img_crops)
    predictions = model.predict(img_crops, verbose=0)

    available = 0
    occupied = 0
    zone_stats = {}

    for i, pos in enumerate(positions):
        x, y, w, h = pos["x"], pos["y"], pos["w"], pos["h"]
        zone = pos.get("zone", "Zona A")

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

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        # Draw label with confidence
        display_text = f"{label} ({confidence:.0%})"
        font_scale = 0.45
        text_thickness = 1
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, text_thickness)[0]
        text_x = x
        text_y = y + h - 5
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 6, text_y + 2), color, -1)
        cv2.putText(img, display_text, (text_x + 3, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                    text_thickness)

        # Zone aggregation (ZAM-01)
        if zone not in zone_stats:
            zone_stats[zone] = {"total": 0, "available": 0, "occupied": 0}
        zone_stats[zone]["total"] += 1
        zone_stats[zone]["available" if is_available else "occupied"] += 1

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

            # Loop video when finished
            if not success:
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
    """Guarda las posiciones definidas desde el SpacePicker."""
    data = request.get_json()
    if not data or 'positions' not in data:
        return jsonify(error="No se recibieron posiciones"), 400

    positions = data['positions']

    # Validate
    for p in positions:
        if not all(k in p for k in ('x', 'y', 'w', 'h', 'zone')):
            return jsonify(error="Formato de posición inválido"), 400

    # Save to pickle
    try:
        with open(POSITIONS_FILE, 'wb') as f:
            pickle.dump(positions, f)
    except Exception as e:
        return jsonify(error=f"Error al guardar: {str(e)}"), 500

    # Update app state
    load_positions()

    return jsonify(success=True, count=len(positions))


@app.route('/load_positions')
def load_positions_route():
    """Carga las posiciones existentes como JSON."""
    return jsonify(positions=app_state["positions"])


# =============================================================================
# Startup
# =============================================================================
# Load positions and initialize video at startup
load_positions()
init_video()

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
