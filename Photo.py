import cv2
import os

save_folder = "KnownFaces"
os.makedirs(save_folder, exist_ok=True)

# Ask for a name
name = input("Enter the name for this face: ").strip()
filename = f"{save_folder}/{name}.jpg"

# Open webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to take photo, ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    cv2.imshow("Capture Face - Press SPACE", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        print("Cancelled.")
        break
    elif key == 32:  # SPACE
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        break

cap.release()
cv2.destroyAllWindows()
