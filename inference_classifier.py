import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb')) # Load the trained model
model = model_dict['model'] # Extract the model

cap = cv2.VideoCapture(0) # Use 0 for web camera, 1 for external camera, 2 for video file/ other cam

mp_hands = mp.solutions.hands # Hands model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # Initialize the Hands model

labels_dict = {0: 'A', 1: 'B', 2: 'L'} # Change this dictionary based on the number of classes you trained your model on
while True: # Run until the user interrupts

    data_aux = [] # List to hold the landmarks for the current frame
    x_ = [] # List to hold x coordinates of landmarks
    y_ = [] # List to hold y coordinates of landmarks

    ret, frame = cap.read() # Read a frame from the camera

    H, W, _ = frame.shape # Get the height and width of the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB for processing by MediaPipe

    results = hands.process(frame_rgb) # Process the frame to find hands
    if results.multi_hand_landmarks: # If hands are found
        for hand_landmarks in results.multi_hand_landmarks: # Loop over each hand found
            mp_drawing.draw_landmarks( 
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks: # Loop over each hand found
            for i in range(len(hand_landmarks.landmark)): # Loop over each landmark
                x = hand_landmarks.landmark[i].x  # Get the x coordinate
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)): # Loop over each landmark again to normalize
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10 # Denormalize the coordinates to get bounding box
        y1 = int(min(y_) * H) - 10 # Denormalize the coordinates to get bounding box

        x2 = int(max(x_) * W) - 10 
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)]) # Make a prediction using the trained model

        predicted_character = labels_dict[int(prediction[0])] # Get the predicted character from the labels dictionary

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # Draw a rectangle around the hand 
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA) # Put the predicted character text above the rectangle

    cv2.imshow('frame', frame) # Show the frame
    cv2.waitKey(1) # Small delay to allow frame to be displayed 


cap.release() # Release the camera
cv2.destroyAllWindows() # Close all OpenCV windows
