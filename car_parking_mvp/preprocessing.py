"""
Módulo de pre-procesamiento y validación de calidad de imagen.
Basado en Use Case CI-02: Validación de resolución y brillo
antes de enviar frames al modelo de detección.
"""
import cv2
import numpy as np


def validate_resolution(frame, min_width=640, min_height=480):
    """Valida que el frame tenga una resolución mínima aceptable."""
    if frame is None:
        return False
    h, w = frame.shape[:2]
    return w >= min_width and h >= min_height


def validate_brightness(frame, min_brightness=30, max_brightness=230):
    """
    Valida que el brillo promedio del frame esté dentro de un rango aceptable.
    Evita procesar imágenes demasiado oscuras (noche sin iluminación)
    o sobreexpuestas (reflejos directos de sol).
    """
    if frame is None:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return min_brightness <= mean_brightness <= max_brightness


def preprocess_frame(frame, target_size=(1280, 720)):
    """
    Pipeline completo de pre-procesamiento:
    1. Valida resolución
    2. Valida brillo
    3. Redimensiona al tamaño objetivo
    
    Returns:
        tuple: (processed_frame, is_valid, issues)
            - processed_frame: frame redimensionado o None si no es válido
            - is_valid: bool indicando si pasó la validación
            - issues: lista de strings describiendo problemas encontrados
    """
    issues = []

    if frame is None:
        return None, False, ["Frame is None"]

    if not validate_resolution(frame):
        issues.append(f"Low resolution: {frame.shape[1]}x{frame.shape[0]}")

    if not validate_brightness(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 30:
            issues.append(f"Too dark (brightness: {brightness:.1f})")
        else:
            issues.append(f"Too bright (brightness: {brightness:.1f})")

    # Even if there are issues, resize and return the frame
    # The caller decides whether to use it based on is_valid
    processed = cv2.resize(frame, target_size)
    is_valid = len(issues) == 0

    return processed, is_valid, issues
