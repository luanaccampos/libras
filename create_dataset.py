import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def get_video_properties(cap: cv2.VideoCapture) -> tuple:
    """
    Gets the video properties
    Args:
        cap: Video capture
    Returns:
        width: Video width
        height: Video height
        fps: Video fps
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return width, height, fps

def write_video(frames: list, output: str, fps: float, width: int, height: int) -> None:
    """
    Writes a video from a list of frames
    Args:
        frames: List of frames
        output: Output video path
        fps: Video fps
        width: Video width
        height: Video height
    """
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

df = pd.DataFrame(columns=[i for i in range(42)])

'''
letra_a = pd.read_csv('letra_a.csv', header=0)
letra_b = pd.read_csv('letra_b.csv', header=0)
letra_r = pd.read_csv('letra_r.csv', header=0)
letra_s = pd.read_csv('letra_s.csv', header=0)
letra_e = pd.read_csv('letra_e.csv', header=0)



Y = [0]*letra_a.shape[0] + [1]*letra_r.shape[0] + [2]*letra_b.shape[0] + [3]*letra_s.shape[0] + [4]*letra_e.shape[0]
df = pd.concat([letra_a, letra_r, letra_b, letra_s, letra_e])
X = df.to_numpy()
neigh = KNeighborsClassifier()
neigh.fit(X, Y)
'''

letras = ['A', 'R', 'B', 'S', 'E']

cap = cv2.VideoCapture(0)
w, h, fps = get_video_properties(cap)
frames = []

with mp_hands.Hands(max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        pts = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = int(hand_landmarks.landmark[i].x * w)
                    y = int(hand_landmarks.landmark[i].y * h)
                    pts.append([x, y])
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        pts = np.array(pts, dtype=float)
        if pts.ndim == 2:
            x1 = int(min(pts[:,0]))
            y1 = int(min(pts[:,1]))
            x2 = int(max(pts[:, 0]))
            y2 = int(max(pts[:, 1]))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            pts[:,0] = [(x-x1)/(x2-x1) for x in pts[:,0]]
            pts[:,1] = [(y-y1)/(y2-y1) for y in pts[:,1]]

            a = np.resize(pts, (1, 42))
            df.loc[len(df.index)] = a[0]

            
            #out = neigh.predict(a)
            #cv2.putText(image, letras[out[0]], (x1-40, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        frames.append(image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
write_video(frames, 'video.mp4', fps, w, h)
df.to_csv('letra_y.csv', index=False)