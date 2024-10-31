import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Screen size
screen_width, screen_height = pyautogui.size()

# Stabilizing cursor movement
stabilizing_points = deque(maxlen=5)  # Queue for recent cursor positions

# Gesture control state
active_control = False

while cap.isOpened():
    # Limit frame processing to every other frame
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 0:
        success, image = cap.read()
        if not success:
            continue

        # Flip the image for a mirrored display
        image = cv2.flip(image, 1)
        
        # Convert BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to find hands
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Use only the first detected hand for control
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get landmarks for index finger tip and thumb tip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Convert hand coordinates to screen coordinates
            screen_x = int(index_tip.x * screen_width)
            screen_y = int(index_tip.y * screen_height)
            
            # Calculate the distance between thumb and index finger
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

            # Check for pinch gesture (distance threshold)
            is_pinching = distance < 0.05

            if is_pinching and not active_control:
                active_control = True
                pyautogui.mouseDown()
            elif not is_pinching and active_control:
                active_control = False
                pyautogui.mouseUp()

            # Add the current cursor position to the smoothing queue
            stabilizing_points.append((screen_x, screen_y))
            avg_x = int(np.average([p[0] for p in stabilizing_points]))
            avg_y = int(np.average([p[1] for p in stabilizing_points]))

            # Move the cursor if the gesture is active
            if active_control:
                pyautogui.moveTo(avg_x, avg_y)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Display gesture status
            cv2.putText(image, 'Gesture Active' if is_pinching else 'Gesture Inactive', 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_pinching else (0, 0, 255), 2)
        
        # Display the image
        cv2.imshow('Virtual Mouse Control', image)
    
    # Break loop with 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
