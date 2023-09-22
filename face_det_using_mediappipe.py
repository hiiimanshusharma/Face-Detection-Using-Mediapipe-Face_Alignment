import cv2
import mediapipe as mp
import numpy as np
import os

# Path to the input video file
video_path = "Chemical - Post Malone (cover by summer).mp4"

# Path to the output video file
output_path = 'video_with_detections_using_mediapipe1.mp4'

# Directory to save cropped frames
output_folder = 'cropped_frames'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize VideoCapture for reading the input video
cap = cv2.VideoCapture(video_path)

# Get video details
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

# Define the codec and create VideoWriter object for writing the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height), isColor=True)

# Define the codec and create VideoWriter object for writing the cropped frames video
cropped_frames_output_path = 'cropped_frames_video.mp4'
cropped_frames_out = cv2.VideoWriter(cropped_frames_output_path, fourcc, frame_rate, (200, 200), isColor=True)  # Change frame size to 200x200

# Create a FaceDetection object
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.95) as face_detection:

    frame_counter = 0  # Counter for naming cropped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop when we reach the end of the video

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with face detection
        results = face_detection.process(frame_rgb)

        # Create a copy of the frame for simple face detection
        frame_with_boxes = frame.copy()

        # Create a black image with the same size as the frame
        black_frame = np.zeros_like(frame)

        # Draw bounding boxes for simple face detection
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
                inside_face = frame[y:y + h, x:x + w]
                black_frame[y:y + h, x:x + w] = inside_face

                # Resize the cropped frame to 200x200
                resized_cropped_frame = cv2.resize(inside_face, (200, 200))

                # Save the resized cropped frame to the output folder
                cropped_frame_filename = os.path.join(output_folder, f'frame_{frame_counter}.jpg')
                cv2.imwrite(cropped_frame_filename, resized_cropped_frame)
                frame_counter += 1

                # Write the cropped frame to the cropped frames video
                cropped_frames_out.write(resized_cropped_frame)

        # Display the frame with simple face detection
        cv2.imshow('Simple Face Detection', frame_with_boxes)

        # Display the frame with inside bounding boxes
        cv2.imshow('Inside Bounding Boxes', black_frame)

        # Write the frame with inside bounding boxes to the output video
        out.write(black_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

# Release video capture and writer objects
cap.release()
out.release()
cropped_frames_out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
