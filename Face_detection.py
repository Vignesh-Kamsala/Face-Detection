import cv2 as cv
import mediapipe as mp

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection

# Set up the video capture
cap = cv.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB for processing
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # Draw bounding boxes if faces are detected
        if results.detections:
            for detection in results.detections:
                # Get the bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)

                # Draw the bounding box
                cv.rectangle(frame, bbox, (0, 255, 0), 2)

        # Display the output
        cv.imshow('Face Detection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv.destroyAllWindows()
