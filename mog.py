import cv2

class MOG:
  def __init__(self, history=100, varThreshold=80, detectShadows=False):
      self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
          history=history,
          varThreshold=varThreshold,
          detectShadows=detectShadows
      )
      self.mask = None

      self.open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
      self.close_kernel = close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))


  def update(self, frame):
    """Updates the subtractor with a new frame"""

    # preprocess the frame
    blurred = cv2.GaussianBlur(frame, (3,3), 0)

    fg_mask = self.background_subtractor.apply(blurred)
    # Apply morphological operations to remove noise and fill holes

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.close_kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.open_kernel)

    self.mask = fg_mask
  
  def draw(self):
     cv2.imshow('MOG', self.mask)

        