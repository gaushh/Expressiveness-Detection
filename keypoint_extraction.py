import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import ConvexHull
import os
def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Function to calculate the openness of a pose
def pose_openness(holistic_landmarks, image, mp_holistic):
    image_h, image_w, _ = image.shape
    keypoints = [
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST],
    ]
    
    core_keypoints = [
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP],
    ]
    # coords = np.array([(kp.x, kp.y) for kp in keypoints])
    coords = np.array([(int(kp.x * image_w), int(kp.y * image_h)) for kp in keypoints])
    hull = ConvexHull(coords)

    core_coords = np.array([(int(kp.x * image_w), int(kp.y * image_h)) for kp in core_keypoints])
    core_hull = ConvexHull(core_coords)
    
    return hull.volume / core_hull.volume

def leaning_direction(holistic_landmarks, mp_holistic):
    nose = holistic_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
    left_hip = holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
    right_hip = holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
    
    avg_hip_z = (left_hip.z + right_hip.z) / 2

    if nose.z < avg_hip_z:
        return "BACKWARD"
    else:
        return "FORWARD"
    
def head_direction(prev, curr, image, mp_holistic):
    image_h, image_w, _ = image.shape
    curr_nose = curr.landmark[mp_holistic.PoseLandmark.NOSE] 
    prev_nose = prev.landmark[mp_holistic.PoseLandmark.NOSE]

    curr_nose_cood = np.array([int(curr_nose.x * image_w), int(curr_nose.y * image_h)])
    prev_nose_cood = np.array([int(prev_nose.x * image_w), int(prev_nose.y * image_h)])
    nose_diff = curr_nose_cood - prev_nose_cood
    horizontal = 'STILL'
    vertical = 'STILL'
    if nose_diff[0] > 0:
        horizontal = "RIGHT"
    elif nose_diff[0] < 0:
        horizontal = 'LEFT'
    
    if nose_diff[1] > 0:
        vertical = 'UP'
    elif nose_diff[1] < 0:
        vertical = 'DOWN'
    return horizontal, vertical

def draw_mouth_landmarks(frame, landmarks):
    # Define the indices for the mouth landmarks (inner and outer lips)
    color = (255, 0, 0)
    thickness = 2
    
    # Draw the landmarks on the frame
    for landmark in landmarks:
        landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(landmark[0], landmark[1], frame.shape[1], frame.shape[0])
        if landmark_px:  
            cv2.circle(frame, landmark_px, 1, color, thickness)
            
def calculate_mouth_curvature(landmarks, frame):
    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

    midpoint_outer_lip = (landmarks_array[61] + landmarks_array[67]) / 2
    midpoint_inner_lip = (landmarks_array[50] + landmarks_array[59]) / 2
    left_corner_mouth = landmarks_array[0]
    right_corner_mouth = landmarks_array[6]
    
#     draw_mouth_landmarks(frame, [landmarks_array[61], landmarks_array[67], landmarks_array[50], landmarks_array[59], landmarks_array[0], landmarks_array[6]])
    draw_mouth_landmarks(frame, landmarks_array[0:19])
    
    # calculate vectors for outer lips
    vector_outer_left = midpoint_outer_lip - left_corner_mouth
    vector_outer_right = midpoint_outer_lip - right_corner_mouth

    # calculate vectors for inner lips
    vector_inner_left = midpoint_inner_lip - left_corner_mouth
    vector_inner_right = midpoint_inner_lip - right_corner_mouth

    # calculate angle using dot product for both corners and both lips
    cos_angle_outer_left = np.dot(vector_outer_left, vector_outer_right) / (np.linalg.norm(vector_outer_left) * np.linalg.norm(vector_outer_right))
    angle_outer = np.arccos(cos_angle_outer_left)

    cos_angle_inner_left = np.dot(vector_inner_left, vector_inner_right) / (np.linalg.norm(vector_inner_left) * np.linalg.norm(vector_inner_right))
    angle_inner = np.arccos(cos_angle_inner_left)

    # convert angles from radians to degrees
    angle_outer = np.degrees(angle_outer)
    angle_inner = np.degrees(angle_inner)
    # Determine if the angle indicates an upward curvature or a smile-like shape
    if angle_inner > 0 and angle_outer > 0:
        is_smile = True
    else:
        is_smile = False

    return is_smile
            

def process_keypoints(video_cut):
    openness = []
    openness_value = 0
    leaning_dir = None
    leaning_forward = 0
    head_horizontal = None
    head_vertical = None 
    is_smile = None
    leaning_backward = 0
    
    head_h_left = 0
    head_h_right = 0
    head_v_up = 0
    head_v_down = 0
    num_smile = 0 
    
    cap = cv2.VideoCapture(video_cut)
    mp_holistic = mp.solutions.holistic
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Get the video dimensions and FPS(input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_interval = round(fps / 10)
    
    # Specify the file path
    video_out = 'results/res.mp4'

    # Remove the file if it exists
    if os.path.exists(video_out):
        os.remove(video_out)

    mp_drawing = mp.solutions.drawing_utils
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # try another codec
    out = cv2.VideoWriter(video_out, fourcc, fps, (1280, 720))  # do not decrease fps
    
    ## Holistic
    prev_landmarks = None
    total_movement = 0
    frame_movement = 0
    holistic_threshold = 0.001  # Adjust the threshold to fine-tune movement detection sensitivity
    holistic_keypoints = [
        mp_holistic.PoseLandmark.LEFT_WRIST,
        mp_holistic.PoseLandmark.RIGHT_WRIST,
        mp_holistic.PoseLandmark.LEFT_ELBOW,
        mp_holistic.PoseLandmark.RIGHT_ELBOW,
    ]

    # Process the video frames
    frame_number = 0
    num_frames_with_person = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame with MediaPipe's Holistic module
        results = holistic.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
        
        frame_number += 1
        
        if frame_number % frame_interval == 0:
            # Draw holistic landmarks on the frame
            if results.pose_landmarks:
                current = results.pose_landmarks

                mp_drawing.draw_landmarks(frame, current, mp_holistic.POSE_CONNECTIONS)
                ## Holistic movements
                # Calculate the total movement
                if prev_landmarks:
                    frame_movement = 0
                    for kp in holistic_keypoints:
                        distance = euclidean_distance(results.pose_landmarks.landmark[kp], prev_landmarks.landmark[kp])
                        frame_movement += distance
                    if frame_movement > holistic_threshold:
                        total_movement += frame_movement
                
                    head_horizontal, head_vertical = head_direction(prev_landmarks, current, frame_rgb, mp_holistic)
                    if head_horizontal == 'RIGHT':
                        head_h_right += 1
                    elif head_horizontal == 'LEFT':
                        head_h_left += 1 

                    if head_vertical == 'UP':
                        head_v_up += 1
                    elif head_vertical == 'DOWN':
                        head_v_down += 1 
                
                openness_value = pose_openness(results.pose_landmarks, frame_rgb, mp_holistic)
                openness.append(openness_value)

                leaning_dir = leaning_direction(results.pose_landmarks, mp_holistic)
                if leaning_dir == 'FORWARD':
                    leaning_forward += 1 
                elif leaning_dir == 'BACKWARD':
                    leaning_backward += 1 
                
                # Check for smile-like shape
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0].landmark
                    is_smile = calculate_mouth_curvature(face_landmarks, frame)
                    if is_smile:
                        num_smile += 1 
                num_frames_with_person += 1
                prev_landmarks = results.pose_landmarks
                
                cv2.putText(frame, f"Cumulative Total Movement: {total_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Openness: {openness_value:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Leaning: {leaning_dir}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(frame, f"Head Horizontal: {head_horizontal}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Head Vertical: {head_vertical}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.putText(frame, f"Is Smile: {is_smile}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "NO PERSON DETECTED", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, prev_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                cv2.putText(frame, f"Cumulative Total Movement: {total_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Openness: {openness_value:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Leaning: {leaning_dir}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)

                cv2.putText(frame, f"Head Horizontal: {head_horizontal}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Head Vertical: {head_vertical}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 0), 2)

                cv2.putText(frame, f"Is Smile: {is_smile}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "NO PERSON DETECTED", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 0), 2)
        out.write(frame)
    values = {}
    if num_frames_with_person != 0:
        values['total_movement'] = total_movement
        values['avg_openness'] = np.mean(openness)
        values['time_leaning_forward'] = leaning_forward / num_frames_with_person
        values['time_leaning_backward'] = leaning_backward / num_frames_with_person
        values['time_head_right'] = head_h_right / num_frames_with_person
        values['time_head_left'] = head_h_left / num_frames_with_person
        values['time_head_up'] = head_v_up / num_frames_with_person
        values['time_head_down'] = head_v_down / num_frames_with_person
        values['time_smile'] = num_smile / num_frames_with_person
    out.release()  
    cap.release()  
    return values