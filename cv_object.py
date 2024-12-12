class Object:
    # Static members for pixel-to-feet transformation
    X_TRANSFORM = (225.5, 786.5, 0.149, 0.149)  # (x1, x2, ft/px1, ft/px2)
    Y_TRANSFORM = (111.5, 178, 0.167, 0.167)    # (y1, y2, ft/px1, ft/px2)

    def __init__(self, bbox=None, object_type=None):
        self.bbox = bbox    # [x0, y0, x1, y1]
        self.object_type = object_type

    def centroid(self):
        """
        Returns centroid of object
        """
        x_centroid = (self.bbox[0] + self.bbox[2]) / 2
        y_centroid = (self.bbox[1] + self.bbox[3]) / 2
        return (x_centroid, y_centroid)
    
    def distance(self, obj):
        """
        Returns distance between centroid of self and obj in pixels
        """
        self_centroid = self.centroid()
        obj_centroid = obj.centroid()
        distance = ((self_centroid[0] - obj_centroid[0]) ** 2 + (self_centroid[1] - obj_centroid[1]) ** 2) ** 0.5
        return distance

    def distance_in_feet(self, obj):
        """
        Returns distance between centroid of self and obj in feet
        """
        self_centroid = self.centroid()
        obj_centroid = obj.centroid()
        
        # Calculate pixel distance
        pixel_distance = self.distance(obj)
        
        # Linear interpolation for X direction
        x1, x2, ft_px1, ft_px2 = self.X_TRANSFORM
        x_transform = ft_px1 + (ft_px2 - ft_px1) * (self_centroid[0] - x1) / (x2 - x1)
        
        # Linear interpolation for Y direction
        y1, y2, ft_px1, ft_px2 = self.Y_TRANSFORM
        y_transform = ft_px1 + (ft_px2 - ft_px1) * (self_centroid[1] - y1) / (y2 - y1)
        
        # Average the transformations
        avg_transform = (x_transform + y_transform) / 2
        
        # Convert pixel distance to feet
        distance_in_feet = pixel_distance * avg_transform
        return distance_in_feet