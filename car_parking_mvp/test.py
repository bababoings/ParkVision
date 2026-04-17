import cv2
import pickle
import numpy as np
from tensorflow.keras.models import  load_model

model = load_model("model_final.h5")

class_dictionary = {0: 'no_car', 1:'car'}

video = cv2.VideoCapture("car_test.mp4")

with open('carposition.pkl', 'rb') as f:
    data = pickle.load(f)
    # Only support new format
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'points' in data[0]:
        positionList = data
    else:
        print("[WARNING] Old .pkl format detected. Please re-define spaces.")
        positionList = []


def crop_quadrilateral(img, points, output_size=(48, 48)):
    """Extract image within a quadrilateral using perspective transform."""
    src_pts = np.array(points, dtype=np.float32)
    w, h = output_size
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped


def checkingCarParking(img):
    imgCrops = []
    spaceCounter = 0

    for pos in positionList:
        points = pos.get("points", [])
        if len(points) != 4:
            continue
        try:
            warped = crop_quadrilateral(img, points)
            imgNormalized = warped / 255.0
            imgCrops.append(imgNormalized)
        except:
            imgCrops.append(np.zeros((48, 48, 3)))

    if len(imgCrops) == 0:
        return

    imgCrops = np.array(imgCrops)
    predictions = model.predict(imgCrops)

    for i, pos in enumerate(positionList):
        points = pos.get("points", [])
        if len(points) != 4:
            continue

        pts_np = np.array(points, dtype=np.int32)
        intId = np.argmax(predictions[i])
        label = class_dictionary[intId]

        if label == 'no_car':
            color = (0, 255, 0)
            thickness = 3
            spaceCounter += 1
            textColor = (0, 0, 0)
        else:
            color = (0, 0, 255)
            thickness = 2
            textColor = (255, 255, 255)

        # Draw quadrilateral
        cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=thickness)

        # Label at centroid
        cx = int(np.mean(pts_np[:, 0]))
        cy = int(np.mean(pts_np[:, 1]))
        font_scale = 0.5
        text_thickness = 1

        # textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        # textX = cx - textSize[0] // 2
        # textY = cy + textSize[1] // 2
        # cv2.rectangle(img, (textX - 3, textY - textSize[1] - 5),
        #               (textX + textSize[0] + 3, textY + 2), color, -1)
        # cv2.putText(img, label, (textX, textY - 3),
        #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, textColor, text_thickness)

    cv2.putText(img, f'Space Count: {spaceCounter}', (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


while True:
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, image = video.read()
    if not ret:
        break
    image = cv2.resize(image, (1280, 720))
    checkingCarParking(image)
    cv2.imshow("Image", image)
    if cv2.waitKey(10) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()