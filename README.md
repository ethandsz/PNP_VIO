# PNP_VIO

A simple Visual–Inertial Odometry (VIO) prototype using marker-based pose estimation via Perspective-n-Point (PnP) and fiducial markers (AprilTags).  

This repository provides a basic framework to compute camera pose in a world frame by detecting known markers (AprilTags) and solving the PnP problem.  

## Overview

- The system detects fiducial markers (AprilTags) in camera images.  
- Known 3D positions of these markers (in a user-defined world frame) are defined in `tags.yaml`.  
- Upon detection, the 2D image corners of the tag are matched to their corresponding 3D coordinates.  
- A PnP solver (e.g. OpenCV’s `solvePnP`) computes the camera’s pose (rotation + translation) relative to the world frame.
- (todo) IMU data — to be integrated for pose estimation when markers temporarily go out of view. Puts the "I" in "VIO"

## Getting Started / Usage

1. **Install and build dependencies**  
   - Install AprilTag.
   - Install CV2
   - Ensure you have a working camera and have performed calibration (intrinsic matrix + distortion coefficients).

2. **Configure `tags.yaml`**  
   - Define each marker’s ID and its 3D pose in your chosen world frame (I chose a corner in the room).  
   - Specify marker size.

3. **Run pose estimation**  
   - Use `run.py` as an entry point. The script reads camera frames, runs fiducial detection, and computes pose via PnP.  
   - Visualize the pose in real-time using the provided `rviz_config.rviz` using rviz2.  
