import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from LeddarTech_sequence_to_csv import save_boxes_to_csv
from LeddarTech_Data_info import COLOR , CLASSES , project_3D_2D
from pioneer.das.api.platform import Platform
import matplotlib.patches as mpatches
from natsort import natsorted

# ========== Parameters ==============

def Visualisation_csv(Data_path,LeddarTech_data_path='/home/watercooledmt/Datasets/Pixset/Raw_Data_exctracted/',Min_px_distance_between_ID = 100,FPS=10):
    # ===================================
    ID_available = False
    Frontal_camera_available = False

    # Check if the data is a csv file:
    if Data_path[-4:] == '.csv':
        csv_path = Data_path
    # Sequence name:
    elif not '/' in Data_path:
        Sequence_name=Data_path
        csv_path = 'GT_LeddarTech_csv/'+Sequence_name+'.csv'
    # sequence path:
    else:
        Sequence_name = Data_path.split('/')[-1]
        csv_path = 'GT_LeddarTech_csv/'+Sequence_name+'.csv'

    # Check if the GT file exists:
    if not os.path.exists(csv_path) and Data_path[:-4] != '.csv':
        print(Data_path[-4:])
        print(' == The file does not exist == \nGenerating GT for ' + Sequence_name)
        save_boxes_to_csv(LeddarTech_data_path+Sequence_name)

    #Check LeddarTech data are available:
    try:
        pf = Platform(LeddarTech_data_path+Sequence_name, progress_bar=False)
        spf = pf.synchronized(['pixell_bfc_box3d-deepen', 'flir_bfc_img-cyl', 'flir_bfc_poly2d-detectron-cyl'],['ouster64_bfc_xyzit'], 2e3)
        print(' == LeddarTech data are available == \n     Adding frontal camera data')
        Frontal_camera_available = True

    except:
        print(' == LeddarTech data are not available == \n')


    df = pd.read_csv(csv_path)

    # Extract all the distinct timestamps
    distinct_timestamps = df['Timestamp'].unique().tolist()

    if 'ID' in df.columns:
        Show_ID = True

    plt.figure(figsize=(15, 15))

    Distinct_Class=[]
    for index_frame, frame in enumerate(distinct_timestamps):
        Data = df[df['Timestamp'] == frame]

        Frame=int(10*frame)


        # Frontal Camera:
        if Frontal_camera_available and 'Z' in Data.columns:
            image_sample = spf[Frame]['flir_bfc_img-cyl']
            image_sample=image_sample.raw
            # Plot the image in the top 1/3 of the plot
            ax1 = plt.subplot(3, 1, 1)
            ax1.imshow(image_sample)

            Transform_matrix = spf[Frame]['ouster64_bfc_xyzit'].compute_transform('flir_bfc')

            # Plot the ID inside the image:
            closest_points = {}
            for index, row in Data.iterrows():
                Class = int(row['Class'])
                ID = int(row['ID'])
                x, y, z = row['X'], row['Y'] , row['Z']
                Projected_points=project_3D_2D(np.array([x, y, z]), Transform_matrix)
                distance = np.sqrt(x**2 + y**2)

                if 0 < Projected_points[0] < image_sample.shape[1] and Projected_points[1]>0 and Projected_points[1]<image_sample.shape[0]:
                    closest_points[index] = {'distance': distance, 'ID': ID, 'Class': Class, 'Projected_points': Projected_points}

            # Sort the closest points by distance
            closest_points = {k: v for k, v in sorted(closest_points.items(), key=lambda item: item[1]['distance'])}

            plotted_points = []
            # Plot each point, ensuring no overlap within 50 pixels
            for point in closest_points.values():
                Projected_points = point['Projected_points']
                too_close = False

                # Check if any already plotted points are within 50 pixels
                for plotted in plotted_points:
                    if np.sqrt((plotted[0]-Projected_points[0])**2 + (plotted[1]-Projected_points[1])**2) < Min_px_distance_between_ID:
                        too_close = True
                        break

                # If no points are too close, plot the current point
                if not too_close:
                    ax1.scatter(Projected_points[0], Projected_points[1], c=np.array(COLOR[point['Class']]) / 255, s=20)
                    ax1.text(Projected_points[0], Projected_points[1], int(point['ID']), fontsize=30, color='white',
                             weight='bold')
                    plotted_points.append(Projected_points)

            ax1.set_title('Frontal Camera frame: ' + str(Frame))
            ax1.axis('off')  # Hide axes for the image

            # Plot the frame graph in the bottom 2/3 of the plot
            ax2 = plt.subplot(3, 1, (2, 3))
        else:
            ax2 = plt.subplot(1, 1, 1)  # Use the entire plot if no camera data

        for index, row in Data.iterrows():
            Class = row['Class']
            x, y = row['X'], row['Y']
            l, w = row['Length'], row['Width']
            rotation = row['Rotation'] + 1e-10
            if Show_ID:
                ID = row['ID']
                # if ID is string print true
                if type(ID) == str:
                    if '_' in ID:
                        ID = ID.split('_')[1]
            # Check if the class has already been plotted
            if Class not in Distinct_Class:
                Distinct_Class.append(int(Class))
            # Compute the coordinates of the corners of the original unrotated box
            half_l, half_w = l / 2, w / 2
            corners = np.array([
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w]
            ])

            # Rotate the corners around the origin
            rotated_corners = np.dot(np.array([[np.cos(rotation), -np.sin(rotation)],
                                               [np.sin(rotation), np.cos(rotation)]]), corners.T).T

            rotated_corners += np.array([x, y])
            # print(x,y)

            # loop the corners to close the rectangle
            rotated_corners = np.vstack((rotated_corners, rotated_corners[0]))

            Class_color = np.array(COLOR[int(Class)]) / 255
            # plot the rotated box
            ax2.plot(-rotated_corners[:, 1], rotated_corners[:, 0], c=Class_color, linewidth=2)
            if Show_ID:
                if x>0 and x<50 and y>-50 and y<50:
                    ax2.text(-y, x, int(ID), fontsize=20,color='black')
            # convert to datetime format
            ax2.set_title('Bird view bounding boxes visualisation')

        # Sort the classes in ascending order
        Distinct_Class = natsorted(Distinct_Class)

        # Create a legend for the classes
        Legend=[]
        for Class in Distinct_Class:
            Class_color=np.array(COLOR[Class]) / 255
            Legend.append(mpatches.Patch(color=Class_color, label=CLASSES[Class]))
        ax2.legend(handles=Legend)
        # x and y equal aspect ratio
        ax2.set_xlim(-50, 50)
        ax2.set_ylim(0, 50)
        ax2.set_aspect('equal', adjustable='box')
        plt.pause(1/FPS)
        plt.clf()

#Visualisation_csv('/home/watercooledmt/PycharmProjects/Smart_intersection_2/ID_Tracking_output/20200610_185206_part1_5095_5195.csv')

#Visualisation_csv('20200611_172353_part5_150_250')