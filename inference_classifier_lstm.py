import cv2
import mediapipe as mp
import numpy as np
import string
from tensorflow.keras.models import load_model

# Load the trained LSTM model
model = load_model('model_lstm.h5')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Setup MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Create the 26-class dictionary
labels_dict = {i: char for i, char in enumerate(string.ascii_uppercase)}

sequence_data = []
sequence_length = 30  # Same as used during training
prediction_threshold = 0.8 # Confidence threshold to display prediction

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    all_hand_data = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)
            
            hand_features = []
            for lm in hand_landmarks.landmark:
                hand_features.append(lm.x - min_x)
                hand_features.append(lm.y - min_y)
            all_hand_data.extend(hand_features)

    # Pad features and add to sequence
    padding_size = 84 - len(all_hand_data)
    all_hand_data.extend([0.0] * padding_size)
    sequence_data.append(all_hand_data)

    # Keep the sequence at the desired length
    if len(sequence_data) > sequence_length:
        sequence_data.pop(0)

    # Make a prediction if the sequence is full
    if len(sequence_data) == sequence_length:
        prediction = model.predict(np.expand_dims(sequence_data, axis=0))
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > prediction_threshold:
            predicted_character = labels_dict.get(predicted_class_index, '?')
            
            # Display the prediction on the frame
            cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
