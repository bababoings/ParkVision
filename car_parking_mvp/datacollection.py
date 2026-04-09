import cv2
import pickle
import numpy as np
import os

# ---- Configuration ----
POSITIONS_FILE = 'carposition.pkl'
IMAGE_FILE = 'car1.png'
POINT_RADIUS = 8  # Visual radius of corner points

# ---- State ----
try:
    with open(POSITIONS_FILE, 'rb') as f:
        data = pickle.load(f)
        # Only load new format
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'points' in data[0]:
            positionList = data
        else:
            print("[WARNING] Old format detected. Starting with empty list.")
            positionList = []
except:
    positionList = []

currentPoints = []  # Points being placed for current quad (0-3 points)


def save_positions():
    """Save positions to pickle in new format."""
    with open(POSITIONS_FILE, 'wb') as f:
        pickle.dump(positionList, f)


def point_in_polygon(px, py, polygon_points):
    """Check if a point is inside a polygon using cv2.pointPolygonTest."""
    pts = np.array(polygon_points, dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (float(px), float(py)), False)
    return result >= 0


def mouseclick(events, x, y, flags, params):
    global currentPoints

    if events == cv2.EVENT_LBUTTONDOWN:
        # Add point to current quad
        currentPoints.append([x, y])

        if len(currentPoints) == 4:
            # Complete quad
            positionList.append({
                "points": [list(p) for p in currentPoints],
                "zone": "Zona A"
            })
            currentPoints = []
            save_positions()
            print(f"[INFO] Quad added. Total: {len(positionList)}")

    if events == cv2.EVENT_RBUTTONDOWN:
        # Remove quad under cursor
        for i, pos in enumerate(positionList):
            if point_in_polygon(x, y, pos["points"]):
                positionList.pop(i)
                save_positions()
                print(f"[INFO] Quad removed. Total: {len(positionList)}")
                break


while True:
    image = cv2.imread(IMAGE_FILE)
    if image is None:
        print(f"[ERROR] Could not load image: {IMAGE_FILE}")
        break
    image = cv2.resize(image, (1280, 720))

    # Draw completed quads
    for idx, pos in enumerate(positionList):
        pts = np.array(pos["points"], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

        # Semi-transparent fill
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 255))
        cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)

        # Corner dots
        for pt in pos["points"]:
            cv2.circle(image, (pt[0], pt[1]), POINT_RADIUS, (255, 0, 255), -1)
            cv2.circle(image, (pt[0], pt[1]), POINT_RADIUS, (255, 255, 255), 1)

        # Label
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = f"{idx + 1}: {pos.get('zone', 'Zona A')}"
        cv2.putText(image, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw in-progress points
    if len(currentPoints) > 0:
        for i, pt in enumerate(currentPoints):
            cv2.circle(image, (pt[0], pt[1]), POINT_RADIUS, (0, 255, 255), -1)
            cv2.circle(image, (pt[0], pt[1]), POINT_RADIUS, (255, 255, 255), 2)
            # Point number
            cv2.putText(image, str(i + 1), (pt[0] - 4, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Lines between placed points
        for i in range(1, len(currentPoints)):
            cv2.line(image, tuple(currentPoints[i - 1]), tuple(currentPoints[i]),
                     (0, 255, 255), 2, cv2.LINE_AA)

    # Instructions
    n = len(currentPoints)
    status = f"Punto {n}/4 | Cajones: {len(positionList)} | Clic Izq: agregar punto | Clic Der: eliminar cajon | Q: salir | ESC: cancelar"
    cv2.putText(image, status, (10, 710),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Space Picker - 4 Point Mode", image)
    cv2.setMouseCallback("Space Picker - 4 Point Mode", mouseclick)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == 27:  # ESC - cancel current points
        currentPoints = []

cv2.destroyAllWindows()