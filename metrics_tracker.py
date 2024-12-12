
class MetricsTracker:
    def __init__(self, max_distance=50, history_size=10, pixels_to_meters=5):
        self.prev_objects = []
        self.cur_objects = []
        self.max_distance = max_distance
        self.mapping = None

        self.history_size = history_size
        self.metrics_history = {
            "speed":[0 for i in range(history_size)],
            "object_count":[0 for i in range(history_size)],
        }
    
    def map_objects(self):
        """
        Creates an index mapping between prev_objects and cur_objects.
        Sets self.mapping to a list of length self.prev_objects with the corresponding index
        in cur_objects if the centroids are close enough together, otherwise -1
        """
        mapping = [-1] * len(self.prev_objects)
        
        for i, prev_obj in enumerate(self.prev_objects):
            min_distance = float('inf')
            min_index = -1
            for j, cur_obj in enumerate(self.cur_objects):
                distance = prev_obj.distance_in_feet(cur_obj)
                if distance <= self.max_distance and distance < min_distance:
                    min_distance = distance
                    min_index = j
            mapping[i] = min_index
        
        self.mapping = mapping
    
    def set_cur_objects(self, new_cur_objects):
        """
        Sets self.cur_objects, updates self.prev_objects, and updates self.mapping
        """
        self.prev_objects = self.cur_objects
        self.cur_objects = new_cur_objects
        self.map_objects()

    def update_speed_history(self):    
        """
        Updates the speed history with the average speed of all objects given the current mapping
        """
        if not self.mapping:
            return 0
        
        total_distance = 0
        count = 0
        
        for i, cur_index in enumerate(self.mapping):
            if cur_index != -1:
                distance = self.prev_objects[i].distance_in_feet(self.cur_objects[cur_index])
                total_distance += distance
                count += 1
        
        if count == 0:
            return 0
        
        average_speed = total_distance / count

        self.metrics_history["speed"].pop(0)
        self.metrics_history["speed"].append(average_speed)
    
    def get_average_speed(self):
        """
        Returns average speed based on history
        """
        return sum(self.metrics_history["speed"]) / self.history_size
    
    def update_object_count_history(self):
        """
        Updates how many objects are on the screen
        """
        current_count = len(self.cur_objects)
        self.metrics_history["object_count"].pop(0)
        self.metrics_history["object_count"].append(current_count)

    def get_average_object_count(self):
        """
        Returns average based on history
        """
        return sum(self.metrics_history["object_count"]) / self.history_size
    
    def update_metrics(self):
        """
        Updates all metrics
        """
        self.update_speed_history()
        self.update_object_count_history()
    
    def get_metrics(self):
        """
        Gets all metrics
        """
        l = [
            self.get_average_speed(),
            self.get_average_object_count(),
        ]
        return '\n'.join([str(i) for i in l])


