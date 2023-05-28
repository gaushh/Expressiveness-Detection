import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import ConvexHull


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
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE],
        holistic_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE],
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


def process_keypoints(video_cut):
    openness = []
    leaning_forward = 0
    leaning_backward = 0

    head_h_left = 0
    head_h_right = 0
    head_v_up = 0
    head_v_down = 0

    cap = cv2.VideoCapture(video_cut)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Get the video dimensions and FPS(input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ## Holistic
    prev_landmarks = None
    total_movement = 0
    frame_movement = 0
    holistic_threshold = 0.001  # Adjust the threshold to fine-tune movement detection sensitivity
    holistic_keypoints = [
        mp_holistic.PoseLandmark.LEFT_WRIST,
        mp_holistic.PoseLandmark.RIGHT_WRIST,
        mp_holistic.PoseLandmark.LEFT_ANKLE,
        mp_holistic.PoseLandmark.RIGHT_ANKLE,
    ]

    # Process the video frames
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame with MediaPipe's Holistic module
        results = holistic.process(frame_rgb)
        # Draw holistic landmarks on the frame
        if results.pose_landmarks:
            current = results.pose_landmarks

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

                # update variables
            prev_landmarks = results.pose_landmarks
        frame_number += 1
        # print(frame_number)

    values = {}
    values['total_movement'] = total_movement
    values['avg_openness'] = np.mean(openness)
    values['time_leaning_forward'] = leaning_forward / frame_number
    values['time_leaning_backward'] = leaning_backward / frame_number
    values['time_head_right'] = head_h_right / frame_number
    values['time_head_left'] = head_h_left / frame_number
    values['time_head_up'] = head_v_up / frame_number
    values['time_head_down'] = head_v_down / frame_number

    return values

# video_cut = '/kaggle/working/video.mp4'
# values = gesture_analysis(video_cut)