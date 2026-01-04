import numpy as np

def cross_track_error(position, path):
    min_dist = float("inf")
    best_idx = 0

    for i in range(len(path) - 1):
        p1 = path[i, :2]
        p2 = path[i + 1, :2]
        seg = p2 - p1

        if np.linalg.norm(seg) < 1e-6:
            d = np.linalg.norm(position - p1)
        else:
            t = np.dot(position - p1, seg) / np.dot(seg, seg)
            t = np.clip(t, 0.0, 1.0)
            proj = p1 + t * seg
            d = np.linalg.norm(position - proj)

        if d < min_dist:
            min_dist = d
            best_idx = i

    return min_dist, best_idx

