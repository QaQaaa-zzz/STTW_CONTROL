"""Archived ridge event definition on contiguous post-control samples (200 Hz)."""
import numpy as np


def first_front_reload(progress, front_force_norm, rear_force_norm, center_s):
    """Find the first front reload and count supported rear samples in its eight-sample window."""
    search=np.flatnonzero(progress>=center_s-.10)
    low=0
    if len(search):
        for k in range(int(search[0]),len(progress)):
            if front_force_norm[k]<1.:
                low+=1
            else:
                if low>=6:
                    rear=rear_force_norm[k:k+8]
                    return k,int(np.count_nonzero(rear>1.)),len(rear)
                low=0
    return None,0,0
