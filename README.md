# Gesture-Controlled-Automatic-Following-Shopping-trolley-using-Deep-learning
This project is a deep learning-powered gesture-controlled smart trolley designed to follow a user hands-free. Built using a Raspberry Pi, Arduino, and ultrasonic sensors, the system recognizes specific hand gestures to control movement and provides real-time audio feedback. It is ideal for use in supermarkets, hospitals, and public spaces, especially for individuals requiring mobility assistance.

**🚀 Features**
✋ Gesture Recognition using Mediapipe and a trained ML model

🛒 Automatic Following mode using ultrasonic sensors

🔊 Audio Feedback for actions like "Follow", "Stop", "Left", and "Right"

🔄 Direction Control using servo motor (left/right) and DC motors (forward/backward)

⚙️ Obstacle Avoidance with three ultrasonic sensors (front, left, right)

🔋 Powered by 12V battery (safe & portable)

🧠 Fully autonomous & patent-filed innovation

**🛠️ Tech Stack**
Raspberry Pi (main controller)

Arduino Uno (motor and sensor control)

Python, OpenCV, Mediapipe for gesture detection

Pickle & Scikit-learn for ML model loading

pygame for audio feedback

L298N Motor Driver, DC motors, Servo Motor

Ultrasonic sensors for obstacle and distance detection

**🔧 Hardware Setup**
Connect 3 ultrasonic sensors to Arduino

Use L298N to control 2 DC motors via Raspberry Pi GPIO

Control servo motor (steering) via PWM (GPIO 18)

Speaker connected via audio jack or USB for voice feedback

12V battery powers motors, sensors, and Raspberry Pi (via buck converter if needed)

**📢 Acknowledgments**
Developed as a final year academic project by Suganeshwaran M. under the guidance of Dr. Kumaran.
Patent filed for the proposed system.
