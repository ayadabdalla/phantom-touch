import numpy as np
def print_stats(pcd):
    print(
        "X-axis: ",
        np.min(np.asarray(pcd.points)[:, 0]),
        np.max(np.asarray(pcd.points)[:, 0]),
        np.mean(np.asarray(pcd.points)[:, 0]),
    )
    print(
        "Y-axis: ",
        np.min(np.asarray(pcd.points)[:, 1]),
        np.max(np.asarray(pcd.points)[:, 1]),
        np.mean(np.asarray(pcd.points)[:, 1]),
    )
    print(
        "Z-axis: ",
        np.min(np.asarray(pcd.points)[:, 2]),
        np.max(np.asarray(pcd.points)[:, 2]),
        np.mean(np.asarray(pcd.points)[:, 2]),
    )