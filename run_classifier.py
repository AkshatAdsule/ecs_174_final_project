import cv2
import sys
from classifier import Classifier
from metrics_tracker import MetricsTracker
from cv_object import Object 
from mog import MOG
import numpy as np

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

     # Approximate camera matrix and distortion coefficients
    mtx = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])  # Example values
    dist = np.array([-0.2, 0.1, 0, 0, 0])  # Example values

    # preprocess the frame
    mog = MOG()

    N_FRAMES = 500
    counter = 0

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_per_sample = length / N_FRAMES

    while cap.isOpened():
        if counter > N_FRAMES:
            break
        if(counter % frames_per_sample != 0):
            counter += 1
            continue
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        mog.update(frame)
        mog.draw()
        counter += 1

    
    cap = cv2.VideoCapture(input_file)

    # Create metrics tracker
    metrics_tracker = MetricsTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        mog.update(frame)
        mog.draw()
        if not ret:
            break

        # Process the frame using the classifier
        bboxes, confidences, class_ids = classifier.post_process(frame, mog.mask)

        # Create object for each bbox
        objects = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            class_id = class_ids[i]
            objects.append(Object(bbox, class_id))
        # Update metrics tracker
        metrics_tracker.set_cur_objects(objects)
        metrics_tracker.update_metrics()
        print(metrics_tracker.get_metrics())

        # Draw the bounding boxes and labels on the frame
        classifier._Classifier__draw(bboxes, confidences, class_ids, frame)

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
