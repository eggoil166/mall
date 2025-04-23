from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class PedestrianTracker:
    def __init__(self, max_age=15, nn_budget=200):
        self.tracker = DeepSort(max_age=max_age, n_init=2, 
                              max_cosine_distance=0.4, nn_budget=nn_budget)
    
    def track_frame(self, frame, detections):
        """Track pedestrians in a single frame."""
        return self.tracker.update_tracks(detections, frame=frame)
    """
    def process_video(self, frames, annotations):
        \"""Process entire video sequence.\"""
        all_tracks = []
        for i, frame in enumerate(frames):
            detections = self._format_detections(annotations[i])
            tracks = self.track_frame(frame, detections)
            all_tracks.append([(track.track_id, track.to_ltrb()) for track in tracks])
        return all_tracks
    """
    
    def _format_detections(self, annotation):
        """Convert annotations to DeepSORT format."""
        return [[[x1, y1, x2-x1, y2-y1], 1.0, 'head'] 
                for (x1, y1), (x2, y2) in annotation]