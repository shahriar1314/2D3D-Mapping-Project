# 2D and 3D Mapping Project

## Overview
This project focuses on processing video data to map the positions of vehicles and pedestrians to a geo-referenced Google Maps image. The implementation unfolds in two primary steps:

1. **Homographies and 2D Processing**
   - Map objects in a video to Google Maps by estimating projective coordinate transformations.
   - Process YOLO object detections and transform them into Google Maps coordinates.

2. **3D Processing** (To be implemented later).

The project uses Python or Octave without any external computer vision libraries (e.g., OpenCV). The outputs include the homography matrix and transformed YOLO detections.

---

## Step 1: Homographies and 2D Processing

### Input Data
1. **Keypoint Matches:**
   - File: `kp_gmaps.mat`
   - Contains Nx4 matrix with N pairs of matching pixel coordinates between the first video frame and the Google Maps image.

2. **Video Sequence:**
   - Image frames named `img_0001.jpg`, `img_0002.jpg`, etc.

3. **YOLO Detections:**
   - Files named `yolo_0001.mat`, `yolo_0002.mat`, etc.
   - Each file includes:
     - `xyxy`: Bounding box coordinates (Mx4 matrix).
     - `id`: Object IDs (vector).
     - `class`: Object classes (vector).

### Output Data
1. **Homography Matrix:**
   - File: `homography.mat`
   - Format: Dictionary containing {`H`: H}, where H is the 3x3 transformation matrix.

2. **Transformed YOLO Detections:**
   - Files named `yolooutput_0001.mat`, `yolooutput_0002.mat`, etc.
   - Format: Same as input `yolo_*.mat`, with updated bounding box coordinates.

3. **(Optional) Transformed Images:**
   - Files named `output_0001.jpg`, `output_0002.jpg`, etc.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/2D3D-Mapping-Project.git
   cd 2D3D-Mapping-Project
