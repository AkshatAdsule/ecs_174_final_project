import cv2
import sys
from classifier import Classifier

def main(input_file):
    # Initialize the classifier
    classifier = Classifier()

    # Open the video file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame using the classifier
        bbox, confidences, class_ids = classifier.post_process(frame)

        # Draw the bounding boxes and labels on the frame
        classifier._Classifier__draw(bbox, confidences, class_ids, frame)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_classifier.py <input_video_file>")
    else:
        main(sys.argv[1])
