import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

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
            
        cv2.putText(frame, 'Ready for Class {}? Press "Q" to START countdown!'.format(j), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Check for 'q' press
        if cv2.waitKey(10) == ord('q'):
            break

    # --- START: 2-Second Countdown ---
    countdown_duration = 2
    start_time = time.time()
    
    while time.time() - start_time < countdown_duration:
        ret, frame = cap.read()
        if not ret:
            break

        remaining_time = countdown_duration - (time.time() - start_time)
        
        cv2.putText(frame, 'Capturing in: {:.1f}s'.format(remaining_time), 
                    (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(10) # Small wait to refresh the window

    # --- START: Image Capture ---
    counter = 0
    print('Starting image capture for class {}'.format(j))
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, 'Class: {} | Images: {}/{}'.format(j, counter + 1, dataset_size), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(1) # Minimal wait to allow display

        # Save the image
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()