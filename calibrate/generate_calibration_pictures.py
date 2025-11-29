import numpy as np
import cv2
#
# W_px, H_px = 2592, 1944
# pixel_size_mm = 0.0014  # 1.4 um
# FOV_H_deg = 90.0
#
# sensor_w = W_px * pixel_size_mm
# sensor_h = H_px * pixel_size_mm
#
# f_mm = sensor_w / (2 * math.tan(math.radians(FOV_H_deg)/2.0))
# fx = f_mm / pixel_size_mm
# fy = fx
# cx, cy = W_px/2.0, H_px/2.0
#
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])
# print("f (mm):", f_mm)
# print("fx,fy (px):", fx, fy)
# print("cx,cy:", cx, cy)
# print("K:\n", K)
#
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30.0
 
cap = cv2.VideoCapture(4)

cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")

num_frames = 0
imgs_saved = 0
while True:
    num_frames += 1

    if num_frames % FPS == 0:
        print(num_frames / FPS)

    ret, frame = cap.read()

    h,  w = frame.shape[:2]

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('s'):
        filename = f"img_{imgs_saved}.jpg"
        cv2.imwrite(f"calibration_pictures/{filename}",frame)
        print(f"Saved as: {filename}")
        imgs_saved += 1


cap.release()
cv2.destroyAllWindows()
