import time
import numpy as np
import cv2

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30.0
FIDUCIAL_MARKER_SIZE = 0.086 #m
 
cap = cv2.VideoCapture(4)

cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


intrinsics = np.array(
 [[404.13447425,   0.,         327.90762183],
 [  0.,         405.21467998, 231.79056369],
 [  0.,           0.,           1.        ]])
	
dist = np.array([[-4.21439573e-01,  2.28135728e-01, 3.68447155e-04, -2.88951890e-04,  -6.37282662e-02]])

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")

num_frames = 0
start_time = time.time()
last_run_time = 0
w, h = FRAME_WIDTH, FRAME_HEIGHT
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist, (w,h), 1, (w,h))

scale_factor = 2.0  

while True:
    num_frames += 1

    if num_frames % FPS == 0:
        print(num_frames / FPS)

    ret, frame = cap.read()

    undistorted_image = cv2.undistort(frame, intrinsics, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = undistorted_image[y:y+h, x:x+w]

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    dst_resized = cv2.resize(dst, (0,0), fx=scale_factor, fy=scale_factor)
    cv2.imshow("Output-undistorted", dst)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

print(f"num frames {num_frames} seconds {time.time() - start_time}")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
