import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import ConvexHull
from scipy.misc import derivative
import os

# Function to process the euclidean distance of body keypoints movement 
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

    coords = np.array([(int(kp.x * image_w), int(kp.y * image_h)) for kp in keypoints])
    hull = ConvexHull(coords)

    core_coords = np.array([(int(kp.x * image_w), int(kp.y * image_h)) for kp in core_keypoints])
    core_hull = ConvexHull(core_coords)
    
    return hull.volume / core_hull.volume

def calculate_expression(landmarks, frame):
    # upper_inner_lip_points = [(landmarks[i].x, landmarks[i].y) for i in [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]]
    lower_inner_lip_points = [(landmarks[i].x, landmarks[i].y) for i in [324, 318, 402, 317, 14, 87, 178, 88, 95, 78]]
    xs, ys = zip(*lower_inner_lip_points)

    # Fit a 3rd degree polynomial to the points
    coefficients = np.polyfit(xs, ys, 3)

    # Calculate the second derivative
    second_derivative = np.polyder(coefficients, m=2)

    # Evaluate the second derivative at the middle point
    middle_x = np.mean(xs)
    curvature = np.polyval(second_derivative, middle_x)

    # Draw the polynomial curve on the frame
    x_values = np.linspace(min(xs), max(xs), 100)
    y_values = np.polyval(coefficients, x_values)

    # Convert normalized coordinates to pixel coordinates
    height, width, _ = frame.shape
    pixel_points = np.array([(x * width, y * height) for x, y in zip(x_values, y_values)], np.int32).reshape((-1, 1, 2))

    # Draw the curve
    cv2.polylines(frame, [pixel_points], isClosed=False, color=(0, 255, 0), thickness=2)

    if curvature < -0.5:
        return "SMILE"
    else:
        return "NEUTRAL"
    
def body_orientation(holistic_landmarks, mp_holistic):
    # Check if the person is facing forward
    left_eye = holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE]
    right_eye = holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE]
    left_elbow = holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
    right_elbow = holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
    nose = holistic_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
   
    if left_elbow.visibility >= 0.5 and right_elbow.visibility >= 0.5 and nose.visibility == 1:
        return 'FORWARD'
    elif left_elbow.visibility < 0.5 and right_elbow.visibility >= 0.5 and nose.visibility == 1:
        return 'LEFT'
    elif right_elbow.visibility < 0.5 and left_elbow.visibility >= 0.5 and nose.visibility == 1:
        return 'RIGHT'
    elif left_elbow.visibility >= 0.5 and right_elbow.visibility >= 0.5 and nose.visibility < 1:
        return 'BACKWARD'
        
def process_keypoints(video_cut):
    # Initialize variables
    frame_number = 0
    num_frames_with_person = 0
    num_frames_with_face = 0

    prev_landmarks = None
    prev_face_landmarks = None 
    prev_frame = None 
    # Output on frame variables 
    total_movement = 0
    frame_movement = 0
    openness_value = 0
    body_o = None 
    facial_e = None
    
    # Output on chart variables 
    openness = []
    body_front = 0
    body_left = 0
    body_right = 0
    body_back = 0 
    num_smile = 0 
    num_sad = 0
    num_neutral = 0
    
    cap = cv2.VideoCapture(video_cut)
    
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh

    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.1, min_tracking_confidence=0.1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.1)
    
    mp_drawing = mp.solutions.drawing_utils
    # Get the video dimensions and FPS(input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = round(fps / 10)
    
    # Specify the file path
    video_out = 'results/res.mp4'
    os.remove(video_out)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # You can also use "XVID" or "MJPG" for AVI files
    out = cv2.VideoWriter(video_out, fourcc, fps, (width, height))
    
    ## Holistic
    holistic_threshold = 0.001  # Adjust the threshold to fine-tune movement detection sensitivity
    holistic_keypoints = [
        mp_holistic.PoseLandmark.LEFT_WRIST,
        mp_holistic.PoseLandmark.RIGHT_WRIST,
        mp_holistic.PoseLandmark.LEFT_ELBOW,
        mp_holistic.PoseLandmark.RIGHT_ELBOW,
    ]

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
                
                # Calculate the total movement
                if prev_landmarks:
                    frame_movement = 0
                    for kp in holistic_keypoints:
                        distance = euclidean_distance(results.pose_landmarks.landmark[kp], prev_landmarks.landmark[kp])
                        frame_movement += distance
                    if frame_movement > holistic_threshold:
                        total_movement += frame_movement

                # Calculate body orientation 
                # body_o = body_orientation(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER], results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER], face_results)
                body_o = body_orientation(results.pose_landmarks, mp_holistic)
                if body_o == 'FORWARD':
                    body_front += 1
                    # Calculate the openness of the pose
                    openness_value = pose_openness(results.pose_landmarks, frame_rgb, mp_holistic)
                    openness.append(openness_value)
                elif body_o == 'LEFT':
                    body_left += 1 
                    openness_value = -1
                elif body_o == 'RIGHT':
                    body_right += 1
                    openness_value = -1
                elif body_o == 'BACKWARD':
                    body_back += 1
                    openness_value = pose_openness(results.pose_landmarks, frame_rgb, mp_holistic)
                    openness.append(openness_value)

                # Calculate for facial expression 
                if face_results.multi_face_landmarks:
                    print('face')
                    print(num_frames_with_face)
                    num_frames_with_face += 1
                    face_landmarks = face_results.multi_face_landmarks[0].landmark
                    facial_e = calculate_expression(face_landmarks, frame)
                    # assign the number of smile, sad, and neutral
                    if facial_e == 'SMILE':
                        num_smile += 1
                    elif facial_e == 'NEUTRAL':
                        num_neutral += 1
                    # prev_face_landmarks = face_results.multi_face_landmarks[0].landmark
                else:
                    facial_e = 'CANNOT DETERMINE FACIAL EXPRESSION'
                    
                num_frames_with_person += 1
                prev_landmarks = results.pose_landmarks
                
                cv2.putText(frame, f"Cumulative Total Movement: {total_movement:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                if openness_value != -1:
                    cv2.putText(frame, f"Current Openness: {openness_value:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                else:
                    cv2.putText(frame, "NO OPENNESS VALUE AVAILABLE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                cv2.putText(frame, f"Current Body Orientation: {body_o}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                cv2.putText(frame, f"Current Facial Expression: {facial_e}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)

                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cv2.putText(frame, f'Landmark {id}: x={lm.x:.2f}, y={lm.y:.2f}, z={lm.z:.2f}, visibility={lm.visibility:.2f}', 
                                (w-600, 20*(id+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "NO PERSON DETECTED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (231, 22, 22), 2)
        else:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, prev_landmarks, mp_holistic.POSE_CONNECTIONS)
                cv2.putText(frame, f"Cumulative Total Movement: {total_movement:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                if openness_value != -1:
                    cv2.putText(frame, f"Current Openness: {openness_value:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                else:
                    cv2.putText(frame, "NO OPENNESS VALUE AVAILABLE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                cv2.putText(frame, f"Current Body Orientation: {body_o}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
                cv2.putText(frame, f"Current Facial Expression: {facial_e}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 61, 231), 2)
            else:
                cv2.putText(frame, "NO PERSON DETECTED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (231, 22, 22), 2)
        out.write(frame)

    values = {}

    if num_frames_with_person != 0:
        values['avg_total_movement'] = total_movement / num_frames_with_person
        values['avg_openness'] = np.mean(openness)
        values['percentage_body_turn_left'] = body_left / num_frames_with_person
        values['percentage_body_turn_right'] = body_right / num_frames_with_person
        values['percentage_body_turn_front'] = body_front / num_frames_with_person
        values['percentage_body_turn_back'] = body_back / num_frames_with_person
    print(num_frames_with_face)
    print(num_smile)
    print(num_neutral)
    if num_frames_with_face != 0:
        values['percentage_smile'] = num_smile / num_frames_with_face
        values['percentage_neutral'] = num_neutral / num_frames_with_face
        # values['percentage_sad'] = num_sad / num_frames_with_face
    else:
        values['percentage_smile'] = 0
        values['percentage_neutral'] = 0
        # values['percentage_sad'] = 0
    out.release()   
    return values