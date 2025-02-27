import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe hands and utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing labeled sign images
DATA_DIR = './data'

# Prepare lists for data and labels
data = []
labels = []

# Iterate through labeled directories
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        # Read and process the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    # Append normalized coordinates
                    x_.append(x)
                    y_.append(y)

            # Normalize landmarks relative to the bounding box
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append processed data and label
            data.append(data_aux)
            labels.append(dir_)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Release Mediapipe resources
hands.close()

print("Data preprocessing and saving completed.")
