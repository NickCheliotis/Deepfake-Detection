
import cv2
import mediapipe as mp
import numpy as np

def extract_mouth_open_ratio(video_path):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


    upper_lip_index = 13  # upper inner lip center
    lower_lip_index = 14  # lower inner lip center
    left_mouth_index = 61 # mouth corner left
    right_mouth_index = 291 # mouth corner right

    cap = cv2.VideoCapture(video_path)
    mouth_open_ratios = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            ih, iw = frame.shape[:2]


            points = [(int(p.x * iw), int(p.y * ih)) for p in landmarks.landmark]
            top_lip = points[upper_lip_index]
            bottom_lip = points[lower_lip_index]
            left_lip = points[left_mouth_index]
            right_lip = points[right_mouth_index]


            vertical_dist = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))


            mouth_width = np.linalg.norm(np.array(left_lip) - np.array(right_lip))
            ratio = vertical_dist / mouth_width if mouth_width > 0 else 0

            mouth_open_ratios.append(ratio)

    cap.release()
    face_mesh.close()

    if len(mouth_open_ratios) == 0:
        return 0

    return np.mean(mouth_open_ratios)

