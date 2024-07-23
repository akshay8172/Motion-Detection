import cv2
import os
import numpy as np
from datetime import datetime

# Define the directory to save images and video
SAVE_DIR = "captured_images"
VIDEO_FILE = "motion_video.mp4"

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Capture initial frame
ret, initial_frame = cap.read()
if not ret:
    print("Error: Could not capture initial frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
initial_gray = cv2.GaussianBlur(initial_gray, (21, 21), 0)

# Variables for motion detection
image_count = 0
motion_images = []

# Function to save images
def save_image(frame, count):
    filename = f"{SAVE_DIR}/image{count}.jpg"
    cv2.imwrite(filename, frame)
    return filename

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20
video_writer = None

# Check OpenCV version
opencv_version = cv2.__version__.split('.')
is_opencv_4 = int(opencv_version[0]) == 4
def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compute the absolute difference between the current frame and initial frame
    frame_diff = cv2.absdiff(initial_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    motion_detected = False
    
    dist = distMap(initial_frame, frame)
    mod = cv2.GaussianBlur(dist, (9, 9), 0)
    # apply thresholding
    _, thresh = cv2.threshold(mod, 100, 255, 0)
    # calculate st dev test
    _, stDev = cv2.meanStdDev(mod)
    print(stDev[0][0])
    if stDev[0][0]>20:
        motion_detected=True
        
    # Save images if motion is detected
    if motion_detected:
        image_count += 1
        filename = save_image(frame, image_count)
        motion_images.append(filename)

        # Initialize video writer if not done
        if video_writer is None:
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, fps, (width, height))

        print(f"Image saved as {filename}")

    # Write the frame to video
    if motion_detected:
        video_writer.write(frame)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Finalize the video if it was created
if video_writer is not None:
    video_writer.release()
    print(f"Video saved as {VIDEO_FILE}")

# Remove motion images after video creation
for filename in motion_images:
    os.remove(filename)
    print(f"Deleted {filename}")

print("Process completed.")
