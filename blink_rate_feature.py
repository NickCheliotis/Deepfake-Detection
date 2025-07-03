


import cv2
import mediapipe as mp
import numpy as np

# Eye aspect ratio threshold for blink detection
BLINK_THRESHOLD = 0.23
MIN_CONSEC_FRAMES = 2


def eye_aspect_ratio(landmarks, eye_indices):

    v1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    v2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    h = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    ear = (v1 + v2) / (2.0 * h)
    return ear


def extract_eye_blink_rate(video_path):
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    blink_count = 0
    consecutive_low_frames = 0

    eye_indices_left = [362, 385, 387, 263, 373, 380]
    eye_indices_right = [33, 160, 158, 133, 153, 144]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:

            landmarks = result.multi_face_landmarks[0]
            ih, iw = frame.shape[:2]
            coords = [(int(p.x * iw), int(p.y * ih)) for p in landmarks.landmark]

            left_ear = eye_aspect_ratio(coords, eye_indices_left)
            right_ear = eye_aspect_ratio(coords, eye_indices_right)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                consecutive_low_frames += 1
            else:
                if consecutive_low_frames >= MIN_CONSEC_FRAMES:
                    blink_count += 1
                consecutive_low_frames = 0

    cap.release()
    face_mesh.close()


    blink_rate = blink_count / (duration_sec / 60)
    return blink_rate


