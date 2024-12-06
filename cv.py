#================
# detection lines
#================
import cv2
import numpy as np
from datetime import datetime
import time

class TrafficDetector:
    def __init__(self, min_area=250, detection_lines=None):
        """
        Initialize the traffic detector with configuration parameters
        
        Args:
            min_area (int): Minimum contour area to be considered a vehicle
            detection_lines (list): List of (y1, y2) positions for counting lines
        """
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=1000,
            varThreshold=50,
            detectShadows=True
        )
        self.min_area = min_area
        self.detection_lines = detection_lines or [(0, 300)]  # Default detection zone
        self.vehicle_count = 0
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0

    def preprocess_frame(self, frame):
        """
        Preprocess the frame for better vehicle detection
        
        Args:
            frame (np.array): Input frame
        Returns:
            np.array: Preprocessed frame
        """
        # Convert to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(blurred)
        
        # Apply morphological operations to remove noise and fill holes
        close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)
        
        cv2.imshow('FG Mask', fg_mask)
        cv2.imshow('background_subtractor', blurred)
        
        return fg_mask

    def detect_vehicles(self, frame):
        """
        Detect and track vehicles in the frame
        
        Args:
            frame (np.array): Input frame
        Returns:
            list: List of detected vehicle bounding boxes and IDs
        """
        # Preprocess the frame
        fg_mask = self.preprocess_frame(frame)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process detected contours
        detected_vehicles = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_area:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            centroid = (int(x + w/2), int(y + h/2))
            
            # Track vehicle
            vehicle_id = self.track_vehicle(centroid)
            
            detected_vehicles.append({
                'id': vehicle_id,
                'bbox': (x, y, w, h),
                'centroid': centroid
            })
        
        return detected_vehicles

    def track_vehicle(self, centroid):
        """
        Track vehicles across frames using simple centroid tracking
        
        Args:
            centroid (tuple): (x, y) position of vehicle centroid
        Returns:
            int: Vehicle ID
        """
        # Check if centroid matches any existing tracked vehicle
        for vehicle_id, points in self.tracked_vehicles.items():
            if points and self._is_same_vehicle(points[-1], centroid):
                points.append(centroid)
                return vehicle_id
        
        # If no match found, create new tracked vehicle
        self.tracked_vehicles[self.next_vehicle_id] = [centroid]
        self.next_vehicle_id += 1
        return self.next_vehicle_id - 1

    def _is_same_vehicle(self, point1, point2, max_distance=50):
        """
        Check if two points likely belong to the same vehicle
        
        Args:
            point1 (tuple): First point (x1, y1)
            point2 (tuple): Second point (x2, y2)
            max_distance (int): Maximum allowed distance between points
        Returns:
            bool: True if points likely belong to same vehicle
        """
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance < max_distance
    def process_frame(self, frame):
        """
        Process a video frame and detect/track vehicles
        
        Args:
            frame (np.array): Input frame
        Returns:
            np.array: Annotated frame
            list: Detected vehicles
        """
        # Detect vehicles
        detected_vehicles = self.detect_vehicles(frame)
        
        # Draw detection lines
        for y1, y2 in self.detection_lines:
            cv2.line(frame, (0, y1), (frame.shape[1], y1), (255, 255, 0), 2)
            cv2.line(frame, (0, y2), (frame.shape[1], y2), (0, 255, 255), 2)
        
        # Draw bounding boxes and track vehicles crossing detection lines
        for vehicle in detected_vehicles:
            x, y, w, h = vehicle['bbox']
            vehicle_id = vehicle['id']
            centroid = vehicle['centroid']
            # Check if the centroid falls inside any detection line
            for y1, y2 in self.detection_lines:
                if centroid[1] < y1 or y2 < centroid[1]:
                    break
                    # Increment vehicle count only if it's within the lines
                else:
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw ID
                    cv2.putText(
                        frame,
                        f"ID: {vehicle_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    break  # Ensure a vehicle is not counted multiple times
                
        
        # Draw vehicle count on the frame
        cv2.putText(
            frame,
            #f"Vehicle Count: {self.vehicle_count}",
            f"Vehicle Count: {len(self.tracked_vehicles)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        
        return frame, detected_vehicles


    def vehicle_cleanup(self, frame):
        """
        Remove vehicles that are no longer in the frame or are crossing the threshold
        """
        frame_height, frame_width = frame.shape[:2]
        to_remove = []
        for vehicle_id, points in self.tracked_vehicles.items():
            if points:
                last_point = points[-1]
                x, y = last_point
                # Check if the vehicle centroid is out of frame
                if x < 0 or x > frame_width or y < 10 or y > frame_height:
                    to_remove.append(vehicle_id)
        for vehicle_id in to_remove:
            del self.tracked_vehicles[vehicle_id]
    


def main():
    """
    Main function to run traffic detection on video feed
    """
    # Initialize video capture (0 for webcam, or provide video file path)
    path = 'video2.mp4'
    cap = cv2.VideoCapture(path)
    
    # Initialize detector
    detector = TrafficDetector(
        min_area=25,
        detection_lines=[(0,360)]  # Adjust these values based on your camera view
    )
    
    # Preprocess for the background subtractor
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, vehicles = detector.process_frame(frame)
    
    cap.release()
    cv2.destroyAllWindows()


    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, vehicles = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Traffic Detection', processed_frame)
        
        # Break loop on 'q' press
        
        key = cv2.waitKey()
        if key & 0xFF == ord('q'):
            break
        
        # time.sleep(1 / 12)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()