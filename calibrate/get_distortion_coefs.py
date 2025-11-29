import numpy as np
import cv2 as cv
import glob

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

SQUARE_SIZE = 0.021


intrinsics = np.array(
[[404.13447425,   0.,         327.90762183],
[  0.,         405.21467998, 231.79056369],
[  0.,           0.,           1.        ]])

dist = np.array([[-4.21439573e-01,  2.28135728e-01, 3.68447155e-04, -2.88951890e-04,  -6.37282662e-02]])

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
print(objp)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(f'calibration_pictures/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(gray.shape[::-1])

    # Find the chess board corners
    ret, corners = cv.findChessboardCornersSB(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, intrinsics, dist)
 
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, intrinsics, dist)
 
        img = draw(img,corners2,imgpts)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)

        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            print('next')
        # cv.waitKey(500)


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("RMS error:", ret)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

img = img
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

cv.destroyAllWindows()
