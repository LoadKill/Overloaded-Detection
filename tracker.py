from sort import Sort
import numpy as np

def init_tracker():
    
    return Sort()

def update_tracks(tracker, detections):

    if detections:
        dets_np = np.array(detections)
    
    else:
        dets_np = np.empty((0, 5))
    
    return tracker.update(dets_np)