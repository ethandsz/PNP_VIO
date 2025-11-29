import time
import math
import numpy as np
import cv2
from apriltag import apriltag
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf_transformations import quaternion_from_matrix, translation_from_matrix

from geometry_msgs.msg import PoseStamped

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30.0

FIDUCIAL_MARKER_SIZE = 0.0872 #m

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2) #B - x
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2) #G - y
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2) #R - z
    return img

class PosePublisher(Node):

    def __init__(self):
        super().__init__('Pose_Publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'tag_pose', 10)
     
        cap = cv2.VideoCapture(4)

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


        objp = np.zeros((4,3), np.float32)

        objp[0] = [-FIDUCIAL_MARKER_SIZE/2, FIDUCIAL_MARKER_SIZE/2, 0] #left bottom
        objp[1] = [FIDUCIAL_MARKER_SIZE/2, FIDUCIAL_MARKER_SIZE/2, 0] #right bottom
        objp[2] = [FIDUCIAL_MARKER_SIZE/2, -FIDUCIAL_MARKER_SIZE/2, 0] #right top
        objp[3] = [-FIDUCIAL_MARKER_SIZE/2, -FIDUCIAL_MARKER_SIZE/2, 0] #left top

        detector = apriltag("tagStandard41h12")
        while True:
            num_frames += 1

            if num_frames % FPS == 0:
                print(num_frames / FPS)

            ret, frame = cap.read()

            h,  w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist, (w,h), 1, (w,h))
            undistorted_image = cv2.undistort(frame, intrinsics, dist, None, newcameramtx)

            x, y, w, h = roi
            # undistorted_image_grey = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            undistorted_image = cv2.undistort(frame, intrinsics, dist, None, newcameramtx)
            undistorted_image = undistorted_image[y:y+h, x:x+w]
            undistorted_image_grey = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = detector.detect(image_gray)

            for i in range(len(detections)):
                #Lots of false positives 
                corners = np.reshape(np.array([detections[i]["lb-rb-rt-lt"]]), (4,1,2)).astype(np.int32)

                corner_detections = detections[i]["lb-rb-rt-lt"]

                # cv2.drawContours(frame, (corners,), -1, (0, 255, 0), 3)
                
                tag_ret,rvecs, tvecs = cv2.solvePnP(objp, corner_detections,intrinsics, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                imgpts, jac = cv2.projectPoints(objp, rvecs, tvecs, intrinsics,dist)
                img = draw(frame,corner_detections,imgpts)
                cv2.imshow('img',img)


                print(f"Rvecs: {rvecs}")
                print(f"tvecs: {tvecs}")

                tag_rotation_matrix, _ = cv2.Rodrigues(rvecs) #returns rotation amtrix, and jacobian(unused)
                print(f"Rotation matrix: {tag_rotation_matrix}")
                
                # object_to_camera_se3= np.eye(4)
                # object_to_camera_se3[:3,:3] = tag_rotation_matrix
                # object_to_camera_se3[:3,3:] = tvecs
                #
                # camera_to_object_se3 = np.linalg.inv(object_to_camera_se3)
                #
                # theta = -math.pi/2  # -90 degrees (clockwise viewed from +Z)
                # R_w_o = np.array([
                #     [math.cos(theta), -math.sin(theta), 0.0],
                #     [math.sin(theta),  math.cos(theta), 0.0],
                #     [0.0,              0.0,             1.0]
                # ])
                #
                # T_wo = np.eye(4)
                # T_wo[:3,:3] = R_w_o
                # T_wo[:3,3]  = np.array([0.0, 0.4, 0.0])
                #
                # camera_in_world_se3 = T_wo@camera_to_object_se3
                #
                # print(f"o->t se3: \n {object_to_camera_se3}")
                # print(20*"-")
                #
                # self.publish(camera_in_world_se3)


            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv2.imshow("Output", frame)
            undistorted_image = cv2.resize(undistorted_image, (640,480), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Output-undistorted-greyscale", undistorted_image_grey)
            cv2.imshow("Output-undistorted", undistorted_image)
            if cv2.waitKey(1) == ord('q'):
                break

        print(f"num frames {num_frames} seconds {time.time() - start_time}")
# When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


    def publish(self, se3):
        #todo: Fix this it is not roll pitch and yaw but rather Rodrigues rotation vector
        quat = quaternion_from_matrix(se3)
        translation = translation_from_matrix(se3)

        x = translation[0] 
        y = translation[1]
        z = translation[2]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  
        pose = msg.pose

        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        self.publisher_.publish(msg)
        


def main(args=None):
    rclpy.init(args=args)

    pose_publisher = PosePublisher()
    rclpy.spin(pose_publisher)
    pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
