import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

df = pd.DataFrame(columns=[i for i in range(42)])

cap = cv2.VideoCapture(0)
with mp_hands.Hands() as hands:
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
    
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
df.to_csv('letra_r.csv', index=False)