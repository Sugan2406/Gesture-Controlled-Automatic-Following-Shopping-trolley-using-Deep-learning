import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame

pygame.mixer.init()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Follow', 1: 'Follow', 2: 'Reverse', 3: 'Reverse', 4: 'Stop', 5: 'Stop', 6: "Right", 7: 'Left'}

# Load audio files
audio_files = {
    'Folloe': pygame.mixer.Sound('./audio/Follow.mp3'),
    'Reverse': pygame.mixer.Sound('./audio/Reverse.mp3'),
    'Stop': pygame.mixer.Sound('./audio/Stop.mp3'),
    'Right': pygame.mixer.Sound('./audio/Right.mp3'),
    'Left': pygame.mixer.Sound('./audio/Left.mp3'),
    
}

# Variable to keep track of whether a sound is currently playing
sound_playing = False
last_predicted_character = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        
        if len(data_aux) != 42:
            warning_message = "Warning: Detected " + str(len(data_aux)) + " features. Expected 42 features."
            cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            if predicted_character in audio_files and not sound_playing and predicted_character != last_predicted_character:
                audio_files[predicted_character].play()
                sound_playing = True
                last_predicted_character = predicted_character

            if sound_playing and not pygame.mixer.get_busy():
                sound_playing = False
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
