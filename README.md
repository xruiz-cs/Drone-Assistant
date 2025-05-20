# Gesture-Controlled Face Tracking System for DJI Tello

This project implements a real-time computer vision interface for the DJI Tello drone using Python. It combines face detection, face recognition, and hand gesture classification to enable gesture-based drone control and user-following behavior.

---

## Features

- Face Detection with OpenCV DNN.
- Face Recognition of known users using Dlib via the `face_recognition` library.
- Hand Gesture Control using MediaPipe Hands.
- Drone Movement and command execution via `djitellopy`.

---

## File Overview

### `Drone.py`
Main controller script for the drone:
- Streams video, detects/tracks faces, recognizes users, interprets hand gestures.
- Commands the drone using `djitellopy`.
- Gesture commands:
  - Flip: drone performs a flip.
  - Land: drone lands.
  - Following: stops following the user.
- Face recognition only runs when a face appears or reappears.
- Includes real-time control and safety features.

**Usage**: Press `ESC` to exit.

---

### `Photo.py`
Captures and saves labeled images of known users:
- Saves images in the `KnownFaces/` directory.
- Used by `Drone.py` for facial recognition.

**Usage**: Press `SPACE` to capture a photo or `ESC` to exit.

---

## Requirements

Install dependencies:

```bash
pip install opencv-python mediapipe face_recognition djitellopy
