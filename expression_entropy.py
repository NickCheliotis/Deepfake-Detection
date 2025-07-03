from deepface import DeepFace
from scipy.stats import entropy
import cv2
import numpy as np

def extract_expression_entropy(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    expression_dists = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        if frame_count % 5 == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion_dict = analysis[0]['emotion']
                total = sum(emotion_dict.values())
                norm_probs = [v / total for v in emotion_dict.values()]
                expression_dists.append(norm_probs)
            except Exception as e:

                pass

        frame_count += 1

    cap.release()

    if not expression_dists:
        return 0.0


    avg_distribution = np.mean(expression_dists, axis=0)


    return entropy(avg_distribution)

