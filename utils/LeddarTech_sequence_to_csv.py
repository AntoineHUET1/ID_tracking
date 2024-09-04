import matplotlib.pyplot as plt
from pioneer.das.api.platform import Platform
import numpy as np
import pandas as pd


def save_boxes_to_csv(Sequence_Path, output_path='GT_LeddarTech_csv', nb_points_boxes=20, score=1):
    """
    Extracts bounding box data from the Sequence, processes it, and saves it as a CSV file.

    Parameters:
        Sequence_Path (str): Path to the sequence containing the bounding box data. (need to be extracted)
        output_path (str): Directory where the output CSV will be saved.
        nb_points_boxes (int): Minimum number of points required inside the box to be saved.
        score (int): Score associated with the saved box data.
    """
    Sequence_Name=Sequence_Path.split('/')[-1]

    pf = Platform(Sequence_Path, progress_bar=False)
    spf = pf.synchronized(['pixell_bfc_box3d-deepen', 'flir_bfc_img-cyl','flir_bfc_poly2d-detectron-cyl'], ['ouster64_bfc_xyzit'], 2e3)

    Save_Box=[]
    for frame in range(len(spf)):

        boxes_sample = spf[frame]['pixell_bfc_box3d-deepen']
        Points_Lidar_GT = spf[frame]['ouster64_bfc_xyzit'].get_point_cloud()
        Timestamp = frame * 0.1

        for i in range(len(boxes_sample.raw['data'])):

            x, y, z = boxes_sample.raw['data']['c'][i]  # x, y, z coordinate of the center of the boxe
            #print('Distance from the sensor:',Distance_from_Sensor)
            l, L, H = boxes_sample.raw['data']['d'][i]  # length, width ,height
            beta, gamma, alpha = boxes_sample.raw['data']['r'][i]  # rotation around x,y,z
            id = boxes_sample.raw['data']['id'][i]  # the object instance unique ID
            classe = boxes_sample.raw['data']['classes'][i]  # The object category number
            Flag = boxes_sample.raw['data']['flags'][i]  # Miscellaneous infos

            # Define the rotation matrices around each axis
            alpha_rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                              [np.sin(alpha), np.cos(alpha), 0],
                                              [0, 0, 1]])
            beta_rotation_matrix = np.array([[1, 0, 0],
                                             [0, np.cos(beta), -np.sin(beta)],
                                             [0, np.sin(beta), np.cos(beta)]])
            gamma_rotation_matrix = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                              [np.sin(gamma), np.cos(gamma), 0],
                                              [0, 0, 1]])

            # Define the rotation matrix
            rotation_matrix = np.dot(alpha_rotation_matrix, np.dot(beta_rotation_matrix, gamma_rotation_matrix))

            # Define the dimensions of the box
            dimensions = np.array([l, L, H])

            # Define the 8 corner points of the box
            corner_points = np.array([[dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2],
                                      [dimensions[0] / 2, dimensions[1] / 2, -dimensions[2] / 2],
                                      [-dimensions[0] / 2, dimensions[1] / 2, -dimensions[2] / 2],
                                      [-dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2],
                                      [dimensions[0] / 2, -dimensions[1] / 2, dimensions[2] / 2],
                                      [dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2],
                                      [-dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2],
                                      [-dimensions[0] / 2, -dimensions[1] / 2, dimensions[2] / 2]])

            # Rotate and translate the corner points to their final position
            corner_points = np.dot(rotation_matrix, corner_points.T).T
            corner_points[:, 0] += x
            corner_points[:, 1] += y
            corner_points[:, 2] += z

            # Save 3D point cloud for each box

            # change of base x,y to x',y',alpha:
            ptsXpYp = np.transpose(
                np.array([Points_Lidar_GT[:, 0] * np.cos(alpha) + Points_Lidar_GT[:, 1] * np.sin(alpha),
                          Points_Lidar_GT[:, 1] * np.cos(alpha) - Points_Lidar_GT[:, 0] * np.sin(alpha)]))
            xp, yp = x * np.cos(alpha) + y * np.sin(alpha), y * np.cos(alpha) - x * np.sin(alpha)

            # Keep Lidar points in the boxe
            keep = np.where(
                (ptsXpYp[:, 0] >= xp - l / 2) & (ptsXpYp[:, 0] <= xp + l / 2) & (ptsXpYp[:, 1] >= yp - L / 2) & (
                        ptsXpYp[:, 1] <= yp + L / 2))
            if len(keep[0]) >= 1:  # Lidar Points Find XY
                Z = np.transpose(Points_Lidar_GT[keep, 2])
                keep2 = np.where((Z >= z - H / 2) & (Z <= z + H / 2))
                if len(keep2[0]) > nb_points_boxes:
                    #Save_Boxe:
                    Save_Box.append([Timestamp,score,x,y,z,l,L,alpha,classe,id])

    # Convert collected data to DataFrame and save as CSV
    df = pd.DataFrame(Save_Box, columns=['Timestamp', 'Score', 'X', 'Y','Z','Length', 'Width', 'Rotation', 'Class', 'ID'])
    # Convert 'Class' and 'ID' columns to integers
    df['Class'] = df['Class'].astype(int)
    df['ID'] = df['ID'].astype(int)
    df.to_csv(output_path+'/'+Sequence_Name+'.csv', index=False)