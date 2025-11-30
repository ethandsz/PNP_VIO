import time
import yaml
import math
import numpy as np
import cv2
from apriltag import apriltag
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf_transformations import quaternion_from_matrix, translation_from_matrix, euler_matrix, quaternion_matrix
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

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
        self.april_tags = {}
        self.setup_tags()
        print(self.april_tags)
        self.map_boundary_publisher = self.create_publisher(Marker, "boundary_marker", 10)
        self.camera_publisher = self.create_publisher(PoseStamped, f'camera_pose', 10) #create publisher for tag id

     
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
            self.publish_map_boundaries()
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


                tag_pos = np.array(self.april_tags[tag_id]["position"])
                tag_quat = quaternion_matrix(self.april_tags[tag_id]["orientation"])


                tag_in_world_se3 = np.eye(4)
                tag_in_world_se3[:3, :3] = tag_quat[:3, :3]
                tag_in_world_se3[:3, 3] = tag_pos



                cam_in_world_frame_se3 = tag_in_world_se3 @ cam_in_tag_frame_se3
                print(cam_in_world_frame_se3)

                self.publish_tag(tag_in_world_se3, tag_id)
                self.publish_camera(cam_in_world_frame_se3)

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


    def setup_tags(self):
        with open("tags.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        for tag_info in cfg["tags"]:
            tag_info["publisher"] = self.create_publisher(PoseStamped, f'tag_{tag_info["id"]}_pose', 10) #create publisher for tag id
            self.april_tags[tag_info["id"]] = tag_info


    def publish_camera(self,se3):
        msg = self.make_pose_msg(se3)

        self.camera_publisher.publish(msg)

    def publish_tag(self, se3, id):
        if(id in self.april_tags):
            publisher = self.april_tags[id]["publisher"]
            print("Got publisher")

            msg = self.make_pose_msg(se3)
            publisher.publish(msg)
        else:
            print("Tag not in dict, did you forget to add to yaml?")

    def make_pose_msg(self, se3):
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
            return msg


    def publish_map_boundaries(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.02  # line thickness

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        #my rooom in meters (3 corners)
        points = [
            (0.0, 0.0),
            (-3.3, 0.0),
            (-3.3, 3.3),
        ]

        for px, py in points:
            p = Point()
            p.x = px
            p.y = py
            p.z = 0.0
            marker.points.append(p)

        self.map_boundary_publisher.publish(marker)

        


def main(args=None):
    rclpy.init(args=args)

    pose_publisher = PosePublisher()
    rclpy.spin(pose_publisher)
    pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
