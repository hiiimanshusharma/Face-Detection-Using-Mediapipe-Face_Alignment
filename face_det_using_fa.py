import face_alignment
import cv2
import numpy as np
import torch

# Initialize the face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

# Open the video capture
cap = cv2.VideoCapture(r"Chemical - Post Malone (cover by summer).mp4")  

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_video_with_face_alignment1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the landmarks predictions
    preds = fa.get_landmarks(rgb_frame)

    if preds is not None:
        for landmarks in preds:
            for point in landmarks:
                x, y = point.astype(int)
                cv2.circle(frame, (x, y), 1, (211, 211, 211), -1)  # Draw light gray dots for landmarks

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Frame with Detections', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and output objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
