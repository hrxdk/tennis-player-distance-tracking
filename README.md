
# 🎾 Tennis Player Movement Calculator

## 📌 Overview

This project processes a tennis match video to:

1. Detect the tennis court and correct its perspective.
2. Identify and track players using a YOLO-based object detection model.
3. Determine when players switch sides using scoreboard changes.
4. Calculate and display the distance each player travels throughout the match.

The result is an annotated video with distances, player IDs, and switching logic clearly shown.

---

## ⚙️ Main Logic Breakdown

### 1. **Court Detection (WeightedTennisCourtDetector)**

* The video frames are analyzed to detect white court lines.
* A *weighted detection algorithm* is used to score line positions based on how consistently they appear across frames.
* The most reliable detected points are stored for use in perspective correction.

**Purpose:**
Ensures that subsequent calculations (like distance) are done on a geometrically correct view of the court.

---

### 2. **Perspective Correction & Homography**

* Once the key court points are found, a **homography transformation** is applied.
* This "flattens" the court, making it appear as a true top-down view.
* This step eliminates perspective distortion, so distances measured on the transformed frame match real-world proportions.

**Purpose:**
Without this, distances measured in pixels would be inaccurate due to camera angle distortion.

---

### 3. **Player Detection (YOLO)**

* Each frame is passed through a **YOLO object detection model** trained for tennis players.
* Only detections in valid player regions (within the court) are kept.
* The model outputs bounding boxes, confidence scores, and class IDs.

**Purpose:**
Accurately locating players is critical for tracking movement and identifying switches.

---

### 4. **Player Tracking**

* A tracking algorithm assigns **unique IDs** to each detected player and maintains them across frames.
* Even if a player is briefly occluded or moves quickly, tracking attempts to keep the same ID.
* The player’s position is represented as a point (usually the midpoint of the bottom of their bounding box).

**Purpose:**
Without tracking, distance calculations would get reset whenever detection IDs change.

---

### 5. **Side Switching Detection**

* The scoreboard in the video is monitored frame-by-frame.
* When changes in the displayed player names/scores are detected, it signals a **side switch**.
* After a switch, tracked player IDs are swapped to maintain consistent labeling (e.g., "Player 1" always refers to the same human, regardless of side).

**Purpose:**
Keeps statistics and distances accurate even when players change court sides.

---

### 6. **Distance Calculation**

* For each tracked player:

  * The algorithm measures the distance between their positions in consecutive frames.
  * Distances are calculated in **real-world units** (e.g., meters) using the perspective-corrected frame.
  * These distances are accumulated over time.

**Purpose:**
Gives an ongoing measurement of how much each player has moved during the match.

---

### 7. **Annotated Output Video**

* Each output frame contains:

  * Player IDs.
  * Real-time distance traveled.
  * Visual confirmation of player positions.
  * "Valid" status for court detection and tracking.
* The annotated video is saved as the final result.

**Purpose:**
Provides a visually clear and informative analysis video that can be reviewed later.

---

## 🛠️ Tech Stack

* **OpenCV** – video processing, court detection, perspective transformation.
* **NumPy** – calculations and transformations.
* **YOLO (Ultralytics)** – player detection.
* **Custom Tracking Logic** – keeps consistent IDs over time.
* **Matplotlib** – visualization/debugging.

---
