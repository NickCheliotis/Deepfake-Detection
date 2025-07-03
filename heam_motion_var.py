import cv2
import mediapipe as mp
import numpy as np

def extract_head_motion_var_yaw_pitch(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    yaw_angles = []
    pitch_angles = []


    landmark_ids = [33, 263, 1, 61, 291, 199]

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=np.float64)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        ih, iw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            image_points = []

            for idx in landmark_ids:
                lm = face_landmarks.landmark[idx]
                image_points.append((lm.x * iw, lm.y * ih))

            image_points = np.array(image_points, dtype=np.float32)

            focal_length = iw
            center = (iw / 2, ih / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, _ = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            if success:

                rmat, _ = cv2.Rodrigues(rotation_vector)

                yaw = np.degrees(np.arctan2(rmat[2, 0], np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)))

                pitch = np.degrees(np.arctan2(-rmat[2, 1], rmat[2, 2]))



                if abs(yaw) <= 90:
                    yaw_angles.append(yaw)
                if abs(pitch) <= 90:
                    pitch_angles.append(pitch)

    cap.release()
    face_mesh.close()


    return np.var(yaw_angles), np.var(pitch_angles)



