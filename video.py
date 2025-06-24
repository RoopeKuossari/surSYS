import cv2
from src.face_recognition import faceRecognitionPipeline

def main() -> None:
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Call the face recognition pipeline
        annotated, _ = faceRecognitionPipeline(frame, path=False)
        cv2.imshow("Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # Run the main function to start the webcam and face recognition
    print("Webcam face recognition started. Press 'q' to exit.")
