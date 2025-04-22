import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import RPi.GPIO as GPIO
import time
import os

# =======================
# GPIO Pin Definitions
# =======================
# DC Motors (L298N Motor Driver)
LEFT_MOTOR_FORWARD = 23
LEFT_MOTOR_REVERSE = 24
RIGHT_MOTOR_FORWARD = 27
RIGHT_MOTOR_REVERSE = 22

# Servo Motor (Steering Control)
SERVO_PIN = 18

# Ultrasonic Sensors
TRIG_FRONT = 20
ECHO_FRONT = 21
TRIG_LEFT = 19
ECHO_LEFT = 26
TRIG_RIGHT = 6
ECHO_RIGHT = 13

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup([LEFT_MOTOR_FORWARD, LEFT_MOTOR_REVERSE, RIGHT_MOTOR_FORWARD, RIGHT_MOTOR_REVERSE], GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup([TRIG_FRONT, TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_FRONT, ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

# Servo Motor Initialization
servo = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz PWM
servo.start(7.5)  # Neutral position

# =======================
# CNN Classifier Definition
# =======================
class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =======================
# Load Model and Label Encoder
# =======================
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
model.eval()

label_encoder = model_dict.get('label_encoder', None)

# Open Camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Gesture Labels
labels_dict = {0: 'Follow', 1: 'Follow', 2: 'Reverse', 3: 'Reverse', 4: 'Stop', 5: 'Stop', 6: "Right", 7: 'Left'}


def move_forward():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_REVERSE, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_REVERSE, GPIO.LOW)
    os.system('espeak "Moving Forward"')

def move_reverse():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_REVERSE, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_REVERSE, GPIO.HIGH)
    os.system('espeak "Moving Reverse"')

def stop_motion():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_REVERSE, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_REVERSE, GPIO.LOW)
    os.system('espeak "Stopping"')

def turn_left():
    servo.ChangeDutyCycle(5)  # Turn left (angle ~45°)
    os.system('espeak "Turning Left"')
    time.sleep(1)
    servo.ChangeDutyCycle(7.5)  # Return to center

def turn_right():
    servo.ChangeDutyCycle(10)  # Turn right (angle ~135°)
    os.system('espeak "Turning Right"')
    time.sleep(1)
    servo.ChangeDutyCycle(7.5)  # Return to center

# =======================
# Measure Distance Using Ultrasonic Sensors
# =======================
def get_distance(TRIG, ECHO):
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time, stop_time = time.time(), time.time()
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    distance = (stop_time - start_time) * 34300 / 2
    return distance


try:
    while True:
        data_aux = []
        x_, y_ = [], []

        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Check Distance to Avoid Obstacles
        front_dist = get_distance(TRIG_FRONT, ECHO_FRONT)
        left_dist = get_distance(TRIG_LEFT, ECHO_LEFT)
        right_dist = get_distance(TRIG_RIGHT, ECHO_RIGHT)

        # Stop if Obstacle Detected
        if front_dist < 20:
            stop_motion()
            os.system('espeak "Obstacle detected in front"')
            continue

        if left_dist < 15:
            turn_right()
            continue

        if right_dist < 15:
            turn_left()
            continue

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

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

            if len(data_aux) == 42:
                # Convert to PyTorch tensor and make prediction
                input_tensor = torch.tensor([data_aux], dtype=torch.float32)
                output = model(input_tensor)
                predicted_label = torch.argmax(output, dim=1).item()
                predicted_character = labels_dict[predicted_label]

                # Execute corresponding action
                if predicted_character == 'Follow':
                    move_forward()
                elif predicted_character == 'Reverse':
                    move_reverse()
                elif predicted_character == 'Stop':
                    stop_motion()
                elif predicted_character == 'Left':
                    turn_left()
                elif predicted_character == 'Right':
                    turn_right()

                # Display Prediction
                cv2.putText(frame, predicted_character, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Invalid Hand Detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display Frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
