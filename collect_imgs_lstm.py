import os
import cv2
import time
import uuid

DATA_DIR = './data_lstm'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
number_of_sequences = 20  # Number of sequences per class
sequence_length = 16      # Number of frames per sequence

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # --- START: Wait for 'Q' press to initiate countdown ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, 'Ready for Class {}? Press "Q" to START!'.format(j), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(10) == ord('q'):
            break

    # --- Loop through all the sequences for the class ---
    for seq_num in range(number_of_sequences):
        sequence_dir = os.path.join(class_dir, str(uuid.uuid4()))
        os.makedirs(sequence_dir, exist_ok=True)
        
        # --- START: 2-Second Countdown before each sequence ---
        countdown_duration = 2
        start_time = time.time()
        
        while time.time() - start_time < countdown_duration:
            ret, frame = cap.read()
            if not ret: break

            remaining_time = countdown_duration - (time.time() - start_time)
            
            cv2.putText(frame, 'Capturing Sequence {} in: {:.1f}s'.format(seq_num + 1, remaining_time), 
                        (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(10)

        # --- START: Sequence Capture ---
        print('Starting capture for sequence {} of class {}'.format(seq_num + 1, j))
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret: break
                
            cv2.putText(frame, 'Class: {} | Seq: {}/{} | Frame: {}/{}'.format(j, seq_num + 1, number_of_sequences, frame_num + 1, sequence_length), 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

            # Save the frame
            cv2.imwrite(os.path.join(sequence_dir, '{}.jpg'.format(frame_num)), frame)

cap.release()
cv2.destroyAllWindows()
