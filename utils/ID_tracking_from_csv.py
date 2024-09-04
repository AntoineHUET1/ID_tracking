import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter

# ID tracking parameters:
max_distance_threshold = 0.1 *80 /3.6 * 2 # 0.1 sec * 80 km/h / 3.6 m/s * over speed factor
MAX_MISSING_FRAMES = 8

def update_kalman_filter(detected_boxes, associations, kf, tracked_indices):
    for det_idx, trk_idx in associations.items():
        z = detected_boxes[det_idx][2:4]
        kf[tracked_indices[trk_idx]].update(z)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def group_by_class(boxes):
    class_boxes = {}
    for box in boxes:
        class_name = int(box[7])
        if class_name not in class_boxes:
            class_boxes[class_name] = []
        class_boxes[class_name].append(box)
    return class_boxes

def assign_tracks(detected_boxes, tracked_boxes,max_distance_threshold):

    cost_matrix = np.zeros((len(detected_boxes), len(tracked_boxes)))
    for i, detected_box in enumerate(detected_boxes):
        for j, tracked_box in enumerate(tracked_boxes):
            cost_matrix[i, j] = euclidean_distance(detected_box[2:4], tracked_box[:2])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    associations = {}
    for i, j in zip(row_ind, col_ind):
        Distance = cost_matrix[i, j]
        if Distance <= max_distance_threshold:
            associations[i] = j

    return associations

def ID_Tracking(kf,class_count,max_distance_threshold,MAX_MISSING_FRAMES,frame,data):

    Frame_Tracked_Objects=[]
    if not kf:
        for index, row in data.iterrows():
            x, y = row['X'], row['Y']
            l, w = row['Length'], row['Width']
            rotation = row['Rotation'] + 1e-10
            class_name = int(row['Class'])

            # Generate unique ID:
            if class_name not in class_count:
                class_count[class_name] = 0

            # Generate unique ID based on class count
            obj_id = class_count[class_name]
            class_count[class_name] += 1

            # Create unique object ID based on class name and object count
            obj_id = f"{class_name}_{obj_id}"

            tracker = KalmanFilter(dim_x=4, dim_z=2)
            tracker.x = np.array([x, y, 0, 0])  # Initial state [x, y, dx, dy]
            tracker.F = np.array([[1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])  # State transition matrix
            tracker.H = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]])  # Measurement function
            tracker.P *= 10  # Initial uncertainty

            tracker.obj_id = obj_id

            kf.append(tracker)
            Frame_Tracked_Objects.append([x,y,l,w,rotation,class_name,obj_id])


        return(kf,class_count,Frame_Tracked_Objects)


    # Predict next state
    for tracker in kf:
        tracker.predict()

    class_boxes = group_by_class(data.values)

    Used_Trackers = []
    New_Trackers = []

    for class_name, boxes in class_boxes.items():


        # Get data and corresponding indices from kf based on class name
        tracked_data = []
        tracked_indices = []
        for index, tracker in enumerate(kf):
            if int(tracker.obj_id.split('_')[0]) == class_name:

                tracked_data.append(tracker.x[:2])
                tracked_indices.append(index)

        # Convert tracked_data to a numpy array
        tracked_data = np.array(tracked_data)

        # Perform frame-to-frame association
        associations = assign_tracks(boxes,tracked_data,max_distance_threshold)

        # Update Kalman filter with the associated detections
        update_kalman_filter(boxes, associations, kf, tracked_indices)


        # Save the associations
        for det_idx, trk_idx in associations.items():
            _,_,x,y,l,w,rotation,class_name=boxes[det_idx]
            obj_id=kf[tracked_indices[trk_idx]].obj_id
            Frame_Tracked_Objects.append([x,y,l,w,rotation,class_name,obj_id])
        data_only = [value for value in associations.values()]
        selected_indices = [tracked_indices[index] for index in data_only]

        Used_Trackers += list(selected_indices)

        # Handle unassociated detections (initialize new trackers)
        unassociated_detections = set(range(len(boxes))) - set(associations.keys())

        for index, box in enumerate([boxes[i] for i in unassociated_detections]):
            # Get datas from the boxes:
            x, y = box[2:4]
            l, w = box[4:6]
            rotation = box[6] + 1e-10
            class_name = int(box[7])

            # Generate unique ID:
            if class_name not in class_count:
                class_count[class_name] = 0

            # Generate unique ID based on class count
            obj_id = class_count[class_name]
            class_count[class_name] += 1

            # Create unique object ID based on class name and object count
            obj_id = f"{class_name}_{obj_id}"

            tracker = KalmanFilter(dim_x=4, dim_z=2)
            tracker.x = np.array([x,y, 0, 0])
            tracker.F = np.array([[1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
            tracker.H = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]])
            tracker.P *= 10

            tracker.obj_id = obj_id

            kf.append(tracker)
            New_Trackers.append(len(kf)-1)
            Frame_Tracked_Objects.append([x,y,l,w,rotation,class_name,obj_id])

    # Remove trackers that are not associated with any detections
    for index, tracker in enumerate(kf):
        if index not in Used_Trackers and index not in New_Trackers:
            tracker.missed_frames += 1
            if tracker.missed_frames > MAX_MISSING_FRAMES:
                kf.pop(index)

    return (kf, class_count,Frame_Tracked_Objects)

def ID_Tracking_to_csv(Data_input,max_distance_threshold=0.1 *80 /3.6 * 2, MAX_MISSING_FRAMES=8):

    df_input = pd.read_csv(Data_input)

    #if Z in df_input.columns delete it
    if 'Z' in df_input.columns:
        df_input.drop('Z', axis=1, inplace=True)
    if 'ID' in df_input.columns:
        df_input.drop('ID', axis=1, inplace=True)

    df = pd.DataFrame(columns=['Timestamp', 'Class', 'X', 'Y', 'Length', 'Width', 'Rotation', 'ID'])

    # Specify the filename for the CSV file
    csv_output = Data_input.split('/')[-1]

    # Extract all the distinct timestamps
    distinct_timestamps = df_input['Timestamp'].unique().tolist()

    # -------------------------------
    # ID Tracking
    # -------------------------------
    # Initialize
    # Dictionary to store the count of objects for each class
    class_count = {}
    kf = []  # List to store Kalman filters for each tracked object

    for idx, timestamp in enumerate(distinct_timestamps):

        data = df_input[df_input['Timestamp'] == timestamp]
        kf, class_count,Frame_Tracked_Objects = ID_Tracking(kf, class_count,max_distance_threshold, MAX_MISSING_FRAMES, idx, data)

        columns = list(zip(*Frame_Tracked_Objects))

        if columns:
            # Save CSV DATA:
            # Repeat timestamp for each row in the data
            num_rows = len(columns[0])
            timestamps = [timestamp] * num_rows

            #print(np.array(columns).T)
            #breakpoint()
            # Create DataFrame for the current iteration's data
            data = {
                'Timestamp': timestamps,
                'Class': columns[5],
                'X': columns[0],
                'Y': columns[1],
                'Length': columns[2],
                'Width': columns[3],
                'Rotation': columns[4],
                'ID':columns[6]
            }
            df_iteration = pd.DataFrame(data)

            # Append the DataFrame for the current iteration to the existing DataFrame
            df = pd.concat([df, df_iteration], ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv('ID_Tracking_output/'+csv_output, index=False)


ID_Tracking_to_csv('/home/watercooledmt/PycharmProjects/Smart_intersection_2/GT_LeddarTech_csv/20200610_185206_part1_5095_5195.csv')