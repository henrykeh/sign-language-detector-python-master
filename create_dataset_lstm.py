import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = './data_lstm'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    for seq_dir_ in os.listdir(class_dir):
        sequence_dir = os.path.join(class_dir, seq_dir_)
        if not os.path.isdir(sequence_dir):
            continue

        sequence_data = []
        # Sort frames by name to ensure correct order
        sorted_frames = sorted(os.listdir(sequence_dir), key=lambda x: int(x.split('.')[0]))

        for img_path in sorted_frames:
            img_full_path = os.path.join(sequence_dir, img_path)
            img = cv2.imread(img_full_path)
            if img is None:
                # Handle missing or corrupted images by padding with zeros
                sequence_data.append([0.0] * 84) 
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            all_hand_data = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    min_x, min_y = min(x_), min(y_)
                    
                    # Normalize and collect features for one hand
                    hand_features = []
                    for lm in hand_landmarks.landmark:
                        hand_features.append(lm.x - min_x)
                        hand_features.append(lm.y - min_y)
                    all_hand_data.extend(hand_features)

            # Pad features if one or no hands are detected
            padding_size = 84 - len(all_hand_data)
            all_hand_data.extend([0.0] * padding_size)
            
            sequence_data.append(all_hand_data)

        data.append(sequence_data)
        labels.append(dir_)

with open('data_lstm.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
