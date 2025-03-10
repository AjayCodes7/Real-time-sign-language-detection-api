import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
from flask import Flask, jsonify
import mss
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app, origins=["http://localhost:9000","https://25pxhl0v-9000.inc1.devtunnels.ms"])

# Load the trained model
model_dict = pickle.load(open('D:\B Tech\Major Project\Sign-Language-Recognition-Model\model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

# local_chars = []
# remote_chars = []
local_chars = ""
remote_chars = ""

local = {"left": 475, "top": 345, "width": 660, "height": 550}
remote = {"left": 1135, "top": 345, "width": 660, "height": 550}


@app.route('/get_predicted_chars', methods=['GET'])
def get_predicted_chars():
    """Returns predicted words as json format"""
    return jsonify({'local_chars': local_chars, 'remote_chars': remote_chars})


# def remove_items_periodically():
#     """To remove the out-timed captions"""
#     global local_chars, remote_chars
#     while True:
#         time.sleep(2)  # Wait for 2 seconds
#         if local_chars:
#             local_chars.pop(0)
#         if remote_chars:
#             remote_chars.pop(0)


def process_video():
    """Captures the live video and predict the sign"""
    global local, remote
    global local_chars, remote_chars
    with mss.mss() as sct:
        while True:
            time.sleep(0.5)
            data_aux = []
            x_ = []
            y_ = []
            local, remote = remote, local
            screenshot = sct.grab(remote)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # H, W, _ = frame.shape
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

                expected_length = model.n_features_in_
                if len(data_aux) < expected_length:
                    data_aux += [0] * (expected_length - len(data_aux))
                elif len(data_aux) > expected_length:
                    data_aux = data_aux[:expected_length]

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
                if remote['left'] == 475:
                    # if local_chars.count(predicted_character) < 1:
                    #     local_chars.append(predicted_character)
                    local_chars = predicted_character
                else:
                    # if remote_chars.count(predicted_character) < 1:
                    #     remote_chars.append(predicted_character)
                    remote_chars = predicted_character
                # if len(local_chars) > 5:
                #     local_chars.pop(0)
                # if len(remote_chars) > 5:
                #     remote_chars.pop(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    hands.close()
    cv2.destroyAllWindows()


# Main
if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False}).start()
    # threading.Thread(target=remove_items_periodically, daemon=True).start()
    process_video()
