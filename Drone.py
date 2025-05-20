import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import os
from djitellopy import Tello
import time


# Load known faces
known_encodings = []
known_names = []

for filename in os.listdir("KnownFaces"):
    image = face_recognition.load_image_file(f"KnownFaces/{filename}")
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]
        known_names.append(name)

# Load drone
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")
drone.streamon()

# Load hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load face DNN
face_net = cv2.dnn.readNetFromCaffe("DNN/deploy.prototxt", "DNN/res10_300x300_ssd_iter_140000.caffemodel")

# Gesture detection logic
def get_label(landmarks):
    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    wrist = landmarks[0]
    thumb_tip, thumb_ip = landmarks[4], landmarks[3]
    index_tip, index_pip = landmarks[8], landmarks[6]
    middle_tip, middle_pip = landmarks[12], landmarks[10]
    ring_tip, ring_pip = landmarks[16], landmarks[14]
    pinky_tip, pinky_pip = landmarks[20], landmarks[18]

    fingers_up = [
        index_tip[1] < index_pip[1],
        middle_tip[1] < middle_pip[1],
        ring_tip[1] < ring_pip[1],
        pinky_tip[1] < pinky_pip[1],
    ]

    if dist(thumb_tip, index_tip) < 40 and all(fingers_up[1:]):
        return "Following"

    if fingers_up[0] and fingers_up[3] and not fingers_up[1] and not fingers_up[2]:
        return "Flip"

    if all(fingers_up[:3]) and dist(thumb_tip, pinky_tip) < 40:
        return "Land"

    return "Unknown"

# Face tracking logics
def track_face(drone, face_box, frame_size):
    frame_w, frame_h = frame_size
    x1, y1, x2, y2 = face_box
    face_center_x = (x1 + x2) // 2
    face_center_y = (y1 + y2) // 2
    face_width = x2 - x1

    offset_x = face_center_x - frame_w // 2
    offset_y = face_center_y - frame_h // 2

    # Thresholds
    move_threshold = 120
    forward_threshold = 150  # face too small = move forward
    back_threshold = 200     # face too big = move back

    lr = 0
    fb = 0
    ud = 0
    yaw = 0

    if abs(offset_x) > move_threshold:
        yaw = 50 if offset_x > 0 else -50

    if abs(offset_y) > move_threshold-20:
        ud = -20 if offset_y > 0 else 20

    if face_width < forward_threshold:
        fb = 25
    elif face_width > back_threshold:  
        fb = -25

    drone.send_rc_control(lr, fb, ud, yaw)

# Face recognition logic
def recognize_face(frame):
    global known_encodings, known_names, recognized_name

    face_location = face_recognition.face_locations(frame)
    face_encs = face_recognition.face_encodings(frame, face_location)

    if face_encs:
        matches = face_recognition.compare_faces(known_encodings, face_encs[0], tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encs[0])
        if matches:
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                recognized_name = known_names[best_match]
            else:
                recognized_name = "Unknown"

# Startup
time.sleep(2) 
drone.takeoff()

recognized_name = "Unknown"
face_visible_last_frame = False
following_enabled = True

last_command_time = 0
gesture_start_time = 0
current_gesture = None

# Main loop
while True:
    frame = drone.get_frame_read().frame
    frame = cv2.resize(frame, (960, 720))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Face detection (DNN)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    face_box = None
    largest_face_area = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            area = (x2 - x1) * (y2 - y1)
            if area > largest_face_area:
                largest_face_area = area
                face_box = (x1, y1, x2, y2)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(rgb_frame, recognized_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Run face recognition only on first appearance or after face was lost
                if recognized_name == "Unknown" or not following_enabled:
                    recognize_face(rgb_frame)
                    drone.send_rc_control(0, 0, 0, 0)

                face_visible_last_frame = True

                # Track the face
                if recognized_name != "Unknown" and following_enabled:
                    track_face(drone, face_box, (w, h))

    if face_box is None:
        face_visible_last_frame = False
        recognized_name = "Unknown"

    # Gesture detection
    results = hands.process(rgb_frame)
    gesture = "None"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(rgb_frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]
            gesture = get_label(landmarks)
            cv2.putText(rgb_frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            now = time.time()

            if gesture == current_gesture:
                if now - gesture_start_time >= 1.0 and now - last_command_time >= 2.0:
                    if gesture == "Following":
                        following_enabled = False
                        drone.send_rc_control(0, 0, 0, 0)
                        print("Stopped following")
                    elif gesture == "Flip":
                        try:
                            drone.flip('f')
                            print("Flip!")
                        except:
                            print("Flip failed")
                    elif gesture == "Land":
                        print("Landing")
                        drone.land()

                    last_command_time = now
                    current_gesture = None
                    gesture_start_time = 0
            else:
                current_gesture = gesture
                gesture_start_time = now

    else:
        cv2.putText(rgb_frame, "No hand detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Drone Vision", rgb_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

drone.land()
drone.streamoff()
drone.end()
cv2.destroyAllWindows()