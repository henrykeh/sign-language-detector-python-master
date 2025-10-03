import pickle

import cv2
import mediapipe as mp
import numpy as np
import string

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Setup MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3) 
# Note: static_image_mode=False is better for video streams

# Create the 26-class dictionary: {0: 'A', 1: 'B', ..., 25: 'Z'}
labels_dict = {i: char for i, char in enumerate(string.ascii_uppercase)}

while True:
    all_hand_data = [] # Feature list (up to 84 features)
    all_x_ = [] # All x coords for the bounding box
    all_y_ = [] # All y coords for the bounding box
    
    ret, frame = cap.read()
    if not ret: break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        
        # 1. Draw landmarks and collect all coordinates for the overall bounding box
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect all x/y for the overall bounding box calculation
            for landmark in hand_landmarks.landmark:
                all_x_.append(landmark.x)
                all_y_.append(landmark.y)

        # 2. Extract and normalize features for each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            data_aux_hand = [] # 42 features for a single hand

            # Find min x/y for that specific hand's normalization
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize and collect features
            min_x = min(x_)
            min_y = min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux_hand.append(landmark.x - min_x)
                data_aux_hand.append(landmark.y - min_y)
            
            all_hand_data.extend(data_aux_hand)

        # 3. Pad the feature vector if only one hand was detected (ensure length 84)
        fixed_feature_length = 84
        if len(all_hand_data) < fixed_feature_length:
            padding_size = fixed_feature_length - len(all_hand_data)
            all_hand_data.extend([0.0] * padding_size)
            
        # 4. Use the complete (and padded) feature vector for prediction
        # The model expects a 2D array, so we pass it as [np.asarray(all_hand_data)]
        prediction = model.predict([np.asarray(all_hand_data)])
        
        # The prediction is a numerical class (0-25), map it to a character
        predicted_class_index = int(prediction[0])
        predicted_character = labels_dict.get(predicted_class_index, '?') 

        # 5. Calculate and draw the bounding box around ALL detected hands
        if all_x_ and all_y_: # Check if any landmarks were detected
            x1 = int(min(all_x_) * W) - 10
            y1 = int(min(all_y_) * H) - 10
            x2 = int(max(all_x_) * W) + 10
            y2 = int(max(all_y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()