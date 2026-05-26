"""Parking-space model inference helpers.

This module intentionally has no Flask or Supabase dependencies. The current
runtime passes app_state in from main.py as a compatibility bridge while the
single-camera app is being untangled.
"""

import os

import cv2
import numpy as np


def load_model_lazy(app_state, model_file, model_fallback):
    """Lazy-load the detection model on first use."""
    if app_state["model"] is not None:
        return app_state["model"]

    try:
        from tensorflow.keras.models import load_model as tf_load_model

        if os.path.exists(model_file):
            print(f"[INFO] Loading model: {model_file}")
            app_state["model"] = tf_load_model(model_file)
            app_state["model_input_size"] = (96, 96)
        elif os.path.exists(model_fallback):
            print(f"[INFO] MobileNetV2 model not found, loading fallback: {model_fallback}")
            app_state["model"] = tf_load_model(model_fallback)
            app_state["model_input_size"] = (48, 48)
            app_state["class_dictionary"] = {0: 'Disponible', 1: 'Ocupado'}
        else:
            print("[WARNING] No model file found. Detection will not work.")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    return app_state["model"]


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

    # Extract raw pixels - no perspective distortion.
    crop = img[new_y:new_y+new_h, new_x:new_x+new_w]

    # Resize to model input size
    resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    return resized


def check_parking_spaces(img, app_state, model_file, model_fallback, space_id_func):
    """
    Analyze parking spaces in the given frame.

    Uses dynamic volume-aware cropping (no destructive perspective warping)
    then the model to predict occupancy. Applies anti-flicker logic
    based on confidence threshold (VPD-01 Alternative).

    Returns:
        tuple: (annotated_img, available_count, occupied_count, zone_stats, spaces_status)
            spaces_status is a list of {space_id, zone_id, is_occupied, confidence}
            for the active layout, suitable for upserting to the parking_spaces table.
    """
    model = load_model_lazy(app_state, model_file, model_fallback)
    positions = app_state["positions"]
    active_layout = app_state.get("active_layout", "default")

    if model is None or len(positions) == 0:
        return img, 0, 0, {}, []

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
    spaces_status = []

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

        # Per-space record (Sub-Story 8: individual space mapping for mobile DB)
        spaces_status.append({
            "space_id": space_id_func(active_layout, i),
            "zone_id": zone,
            "is_occupied": not is_available,
            "confidence": round(confidence, 4),
        })

    # Calculate availability confidence per zone (ratio of available spaces to total)
    for zone, stats in zone_stats.items():
        if stats["total"] > 0:
            av_confidence = stats["available"] / stats["total"]
            stats["confidence"] = round(av_confidence, 2)
        else:
            stats["confidence"] = 0.0

    return img, available, occupied, zone_stats, spaces_status
