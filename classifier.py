import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture and Mediapipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

predicted_chars = []
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

        # Ensure data_aux has the correct length by padding with zeros
        expected_length = model.n_features_in_
        if len(data_aux) < expected_length:
            data_aux += [0] * (expected_length - len(data_aux))
        elif len(data_aux) > expected_length:
            data_aux = data_aux[:expected_length]

        # Make a prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]
        if predicted_chars:
            if predicted_chars.count(predicted_character) < 1:
                predicted_chars.append(predicted_character)
        else:
            predicted_chars.append(predicted_character)
        if len(predicted_chars) > 10:
            predicted_chars.pop(0)

    # Display the caption below the video
    caption_text = ' '.join(predicted_chars)
    cv2.putText(frame, caption_text, (50, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
