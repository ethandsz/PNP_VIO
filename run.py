import time
import math
import numpy as np
import cv2
from apriltag import apriltag
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf_transformations import quaternion_from_matrix, translation_from_matrix, euler_matrix

from geometry_msgs.msg import PoseStamped

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30.0

FIDUCIAL_MARKER_SIZE = 0.0872 #m

#draw axis on tag
def draw(img, corners, imgpts):
    placement = (corners[0] + corners[2])/2 #take diagonal corners and find center (left bottom and right top)
    corner = tuple(placement.ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2) #B - x
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2) #G - y
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2) #R - z
    return img

class PosePublisher(Node):

    def __init__(self):
        super().__init__('Pose_Publisher')
        self.tag_publishers = {}
     
        cap = cv2.VideoCapture(0)

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

        objp = np.zeros((4,3), np.float32)

        objp[0] = [-FIDUCIAL_MARKER_SIZE/2, FIDUCIAL_MARKER_SIZE/2, 0] #left bottom
        objp[1] = [FIDUCIAL_MARKER_SIZE/2, FIDUCIAL_MARKER_SIZE/2, 0] #right bottom
        objp[2] = [FIDUCIAL_MARKER_SIZE/2, -FIDUCIAL_MARKER_SIZE/2, 0] #right top
        objp[3] = [-FIDUCIAL_MARKER_SIZE/2, -FIDUCIAL_MARKER_SIZE/2, 0] #left top

        axis = np.float32([[FIDUCIAL_MARKER_SIZE/2,0,0], [0,-FIDUCIAL_MARKER_SIZE/2,0], [0,0,-FIDUCIAL_MARKER_SIZE/2]]).reshape(-1,3)

        detector = apriltag("tagStandard41h12")
        while True:
            num_frames += 1

            if num_frames % FPS == 0:
                print(num_frames / FPS)

            ret, frame = cap.read()

            # h,  w = frame.shape[:2]
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist, (w,h), 1, (w,h))
            # undistorted_image = cv2.undistort(frame, intrinsics, dist, None, newcameramtx)
            # x, y, w, h = roi
            # undistorted_image_grey = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            # undistorted_image = cv2.undistort(frame, intrinsics, dist, None, newcameramtx)
            # undistorted_image = undistorted_image[y:y+h, x:x+w]
            # undistorted_image_grey = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = detector.detect(image_gray)

            for i in range(len(detections)):
                print(detections[i])
                tag_id = detections[i]["id"]
                corners = np.reshape(np.array([detections[i]["lb-rb-rt-lt"]]), (4,1,2)).astype(np.int32)
                cv2.drawContours(frame, (corners,), -1, (0, 255, 0), 3) #bouding box

                corner_detections = np.array(detections[i]["lb-rb-rt-lt"], dtype=np.float32)

                
                _, rvecs, tvecs = cv2.solvePnP(objp, corner_detections,intrinsics, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, intrinsics,dist)
                draw(frame,corner_detections,imgpts)

                # print(f"Rvecs: {rvecs}")
                # print(f"tvecs: {tvecs}")

                tag_rotation_matrix, _ = cv2.Rodrigues(rvecs) #returns rotation 3x3 amtrix, and jacobian(unused)
                # print(f"Rotation matrix: {tag_rotation_matrix}")
                
                tag_in_cam_frame_se3= np.eye(4)
                tag_in_cam_frame_se3[:3,:3] = tag_rotation_matrix
                tag_in_cam_frame_se3[:3,3:] = tvecs

                cam_in_tag_frame_se3 = np.eye(4)
                cam_in_tag_frame_se3[:3, :3] = tag_rotation_matrix.T 
                cam_in_tag_frame_se3[:3, 3:] = -tag_rotation_matrix.T @ tvecs

                # tag_in_world_frame_se3 = np.eye(4)
                # r, p, y = -math.pi/2, 0.0, 0.0
                # T = euler_matrix(r, p, y)
                # R = T[0:3, 0:3]
                # tag_in_world_frame_se3[:3,:3] = R
                # tag_in_world_frame_se3[:3,3:] = np.vstack(np.array([-0.4,0.0,0.0]))
                #

                R = np.array([
                    [0,  0, 1],
                    [-1,  0,  0],
                    [0, -1,  0]
                ], dtype=float)

                tag_in_world_se3 = np.eye(4)
                tag_in_world_se3[:3, :3] = R
                tag_in_world_se3[:3, 3] = np.array([0.4, 0.0, 0.0])  # set z here if needed

                cam_in_world_frame_se3 = tag_in_world_se3 @ cam_in_tag_frame_se3
                print(cam_in_world_frame_se3)

                self.publish_tag(cam_in_world_frame_se3, tag_id)

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        print(f"num frames {num_frames} seconds {time.time() - start_time}")
# When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


    def publish_tag(self, se3, id):
        publisher = None
        print(id)
        if(id in self.tag_publishers):
            publisher = self.tag_publishers[id]
            print("Got publisher")
        else:
            self.tag_publishers[id] = self.create_publisher(PoseStamped, f'tag_{id}_pose', 10)
            publisher = self.tag_publishers[id]
            print("Created")

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

        publisher.publish(msg)

        


def main(args=None):
    rclpy.init(args=args)

    pose_publisher = PosePublisher()
    rclpy.spin(pose_publisher)
    pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
