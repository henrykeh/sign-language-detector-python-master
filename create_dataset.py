import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    if not os.path.isdir(dir_path):
        continue
        
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None: continue # Skip if image read fails
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            
            # Initialize a list of all features for up to 2 hands (84 features total)
            all_hand_data = [] 
            
            # Sort hands to maintain a consistent order if possible (e.g., left-to-right on the image)
            # This is complex and often skipped; we'll simply use the order returned by MediaPipe.
            
            # 1. Extract and normalize landmarks for each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux_hand = []

                # Find min x/y for normalization
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize and collect 42 features (21 x 2)
                min_x = min(x_)
                min_y = min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux_hand.append(landmark.x - min_x)
                    data_aux_hand.append(landmark.y - min_y)
                
                all_hand_data.extend(data_aux_hand)

            # 2. Pad the feature list if only one hand was detected
            # Each hand contributes 42 features (21 landmarks * 2 coords).
            # If len(results.multi_hand_landmarks) == 1, len(all_hand_data) == 42
            # If len(results.multi_hand_landmarks) == 2, len(all_hand_data) == 84
            if len(all_hand_data) < 84:
                 # Pad with zeros to ensure fixed size (42 features missing)
                padding_size = 84 - len(all_hand_data)
                all_hand_data.extend([0.0] * padding_size)
            
            # The final feature vector for the image is all_hand_data (guaranteed length 84)
            data.append(all_hand_data) 
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()