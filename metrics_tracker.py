class MetricsTracker:
    DT = 1/30
    FPS_TO_MPH = 1/(1.467)
    def __init__(self, max_distance=7, history_size=10, pixels_to_meters=5):
        self.prev_objects = []
        self.cur_objects = []
        self.max_distance = max_distance
        self.mapping = None

        self.history_size = history_size
        class_indexes = [0, 1, 2, 3, 5, 7, 8]
        self.metrics_history = {
            "speed": {i: [0 for _ in range(history_size)] for i in class_indexes},
            "object_count": {i: [0 for _ in range(history_size)] for i in class_indexes},
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
        
        class_indexes = [0, 1, 2, 3, 5, 7, 8]
        total_distance_by_class = {i: 0 for i in class_indexes}
        count_by_class = {i: 0 for i in class_indexes}
        
        for i, cur_index in enumerate(self.mapping):
            if cur_index != -1:
                distance = self.prev_objects[i].distance_in_feet(self.cur_objects[cur_index])
                class_id = self.prev_objects[i].object_type
                total_distance_by_class[class_id] += distance
                count_by_class[class_id] += 1
        
        for class_id in total_distance_by_class:
            if count_by_class[class_id] > 0:
                average_speed = ((total_distance_by_class[class_id] / count_by_class[class_id]) / self.DT) * self.FPS_TO_MPH
            else:
                average_speed = 0
            self.metrics_history["speed"][class_id].pop(0)
            self.metrics_history["speed"][class_id].append(average_speed)
    
    def get_average_speed(self):
        """
        Returns average speed based on history for each class
        """
        class_indexes = [0, 1, 2, 3, 5, 7, 8]
        average_speed_by_class = {i: sum(self.metrics_history["speed"][i]) / self.history_size for i in class_indexes}
        return average_speed_by_class
    
    def update_object_count_history(self):
        """
        Updates how many objects are on the screen per class
        """
        class_indexes = [0, 1, 2, 3, 5, 7, 8]
        count_by_class = {i: 0 for i in class_indexes}
        for obj in self.cur_objects:
            count_by_class[obj.object_type] += 1
        
        for class_id in count_by_class:
            self.metrics_history["object_count"][class_id].pop(0)
            self.metrics_history["object_count"][class_id].append(count_by_class[class_id])
        
        print(count_by_class)

    def get_average_object_count(self):
        """
        Returns average object count based on history for each class
        """
        class_indexes = [0, 1, 2, 3, 5, 7, 8]
        average_count_by_class = {i: sum(self.metrics_history["object_count"][i]) / self.history_size for i in class_indexes}
        return average_count_by_class
    
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
        average_speed_by_class = self.get_average_speed()
        average_object_count = self.get_average_object_count()
        metrics = {
            "average_speed_by_class": average_speed_by_class,
            "object_count": average_object_count,
        }
        return metrics


