import cv2
from src.face_recognition import faceRecognitionPipeline

def run_webcam():
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam face recognition started. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Call the face recognition pipeline
        annotated_frame, _ = faceRecognitionPipeline(frame, path=False)

        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()  # Run the main function to start the webcam and face recognition
