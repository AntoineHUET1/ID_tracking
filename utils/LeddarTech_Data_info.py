from pioneer.common import linalg
import numpy as np

CLASSES = [
    'pedestrian', 'deformed pedestrian', 'bicycle', 'car', 'van', 'bus', 'truck',
    'motorcycle', 'stop sign', 'traffic light', 'traffic sign', 'traffic cone', 'fire hydrant',
    'guard rail', 'pole', 'pole group', 'road', 'sidewalk', 'wall', 'building', 'vegetation',
    'terrain', 'ground', 'crosstalk', 'noise', 'others', 'animal', 'unpainted', 'cyclist',
    'motorcyclist', 'unclassified vehicle', 'obstacle', 'trailer', 'barrier', 'bicycle rack',
    'construction vehicle'
]

COLOR = [
    (176, 242, 182), (9, 82, 40), (255, 127, 0), (119, 181, 254), (15, 5, 107), (206, 206, 206),
    (91, 60, 17), (88, 41, 0), (217, 33, 33), (255, 215, 0), (48, 25, 212), (230, 110, 60),
    (240, 0, 32), (140, 120, 130), (80, 120, 130), (80, 120, 180), (30, 30, 30), (30, 70, 30),
    (230, 230, 130), (230, 130, 130), (60, 250, 60), (100, 140, 40), (100, 40, 40), (250, 10, 10),
    (250, 250, 250), (128, 128, 128), (250, 250, 10), (255, 255, 255), (198, 238, 242),
    (100, 152, 255), (50, 130, 200), (100, 200, 50), (255, 150, 120), (100, 190, 240),
    (20, 90, 200), (80, 40, 0), (128, 128, 128)
]

def project_3D_2D(point, Transform_matrix):

    point_cloud_in_camera_ref = linalg.map_points(Transform_matrix,point)

    # Manual Method (Cylindrical projection) => same result
    pts2=point_cloud_in_camera_ref.T
    azimut = np.arctan2(pts2[0], pts2[2])
    norm_xz = np.linalg.norm(pts2[[0, 2]], axis=0)
    elevation = np.arctan2(pts2[1], norm_xz)
    image_width = 2000
    image_height = 500
    FOV_width = 3.6651914291880923
    FOV_height = 1.1780972450961724

    x = int(np.round(image_width / 2 + azimut * (image_width / FOV_width)))
    y = int(np.round(image_height / 2 + elevation * (image_height / FOV_height)))

    pts2d = [x, y]
    return pts2d