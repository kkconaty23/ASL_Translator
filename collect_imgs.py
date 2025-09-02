import os # For creating directories and handling file paths

import cv2 # OpenCV library for image processing


DATA_DIR = './data' # Directory where the data will be stored. Change this path if needed.
if not os.path.exists(DATA_DIR): # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR)

number_of_classes = 3 # Change this value based on the number of classes you want to collect data for (ie 26 for alphabet)
dataset_size = 100 # Number of images per class

cap = cv2.VideoCapture(0) # Use 0 for web camera, 1 for external camera, 2 for video file/ other cam
for j in range(number_of_classes): # Loop over the number of classes
    if not os.path.exists(os.path.join(DATA_DIR, str(j))): # Create a directory for each class
        os.makedirs(os.path.join(DATA_DIR, str(j))) # Create directory for class j

    print('Collecting data for class {}'.format(j))

    done = False
    while True: # Wait until user is ready
        ret, frame = cap.read() # Read a frame from the camera
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA) # Display instructions on the frame
        cv2.imshow('frame', frame) # Show the frame
        if cv2.waitKey(25) == ord('q'): 
            break

    counter = 0 # Initialize counter for images collected
    while counter < dataset_size: # Collect images until we reach the dataset size
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25) # Small delay to allow frame to be displayed
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release() # Release the camera
cv2.destroyAllWindows() # Close all OpenCV windows
