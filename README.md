# 🎾 Tennis Ball 3D Trajectory & Speed Estimation

This project reconstructs the **3D trajectory** of a tennis ball using multi-camera video data, estimates the **ball speed** and **bounce events**, and validates the results with **reprojection error metrics** and **gravity acceleration fitting**.  
It is designed as a research project combining **computer vision, camera calibration, triangulation (DLT)**, and **physics-based analysis**.

---

## 🛠 Methods Used
- **Camera Calibration:** Intrinsic and extrinsic parameters are obtained from XML calibration files.
- **Undistortion:** Raw 2D points are corrected for lens distortion using calibration parameters.
- **Triangulation (DLT):** 2D points from multiple calibrated cameras are reconstructed into 3D coordinates.
- **Trajectory Analysis:** 3D points are segmented to detect bounce events and velocity changes.
- **Speed Estimation:** Ball speed is calculated from frame-to-frame displacement, using **24 FPS** as a reference.
- **Gravity Fitting:** Vertical motion (Z-axis) is fitted to estimate gravitational acceleration.
- **Reprojection Error Analysis:** 3D points are projected back into 2D to measure calibration and triangulation accuracy (**pixel → cm** error check).

---

## 🚀 Usage Guide

### 1) Obtain Ball Labels
- Use **CVAT** or another annotation tool to label the tennis ball frame by frame.  
- Export the annotations as **XML files**.

### 2) Convert XML to TXT,

python xmltotxtv2.py

### 3) Undistort Points,
python undistort_points_v2.py

### 4) Triangulation (3D Reconstruction)

## For 3 cameras:

python triangulate_dlt_v5.py


##  For 2 cameras only:

python triangulate_dlt_with2cameras.py


Outputs:

triangulated_points_*.txt

### 5) Visualization

Static 3D visualization

python visualize_3d_points_manual.py


### → Produces a scatter plot of the full 3D trajectory.

Animated trajectory visualization

python animate_3d_points_v2.py


### → Produces an animated trajectory segment (time evolution).

### 6) Speed & Bounce Analysis

Smoothed speed analysis (24 FPS):

python plot_speed.py


Raw speed analysis (24 FPS, unsmoothed):

python speed_alaysis_part_auto_detection.py

### 7) Validation & Extra Tools

Check FPS of video

python fps_kontrol.py


### Gravity fitting (estimate g)

python gravity_fit_v5.py


### Reprojection error & pixel → cm scaling

python pixel_error_control.py

### Example Outputs

3D trajectory visualization (scatter plot of triangulated points)

<img width="720" height="564" alt="image" src="https://github.com/user-attachments/assets/b2870de8-ff7c-44fe-a785-8ce96c5e65fe" />


Animated trajectory (segment-by-segment motion)

<img width="671" height="473" alt="image" src="https://github.com/user-attachments/assets/7f625f1b-7be5-4a30-b245-38bbca599116" />

Speed graphs with bounce detection points

<img width="634" height="757" alt="image" src="https://github.com/user-attachments/assets/00208375-43af-4889-9e71-ffc765a6782c" />


Reprojection error statistics in pixels and centimeters

<img width="595" height="262" alt="image" src="https://github.com/user-attachments/assets/9a86a244-5427-40f7-86a2-b37b4ed21e2f" />


### Repository Structure
configuration_files/        # Camera calibration and config
xml_files/                  # Raw XML annotation files
undistorted_points_v2/      # Lens-corrected points
triangulated_points_*.txt   # 3D reconstructed points
*.py                        # Python scripts for each step

###  Applications

Sports analytics (tennis, football, etc.)

Physics experiments (projectile motion, gravity estimation)

Multi-camera 3D reconstruction research
