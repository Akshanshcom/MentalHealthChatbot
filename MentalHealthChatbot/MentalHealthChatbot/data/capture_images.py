import cv2
import os

# Define the expressions you want to capture
expressions = ['happy', 'sad', 'angry', 'surprised', 'neutral']

# Create directories for each expression
base_dir = 'data/processed/images'
for expression in expressions:
    expression_dir = os.path.join(base_dir, expression)
    if not os.path.exists(expression_dir):
        os.makedirs(expression_dir)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

current_expression_index = 0
capture_mode = False

while current_expression_index < len(expressions):
    expression = expressions[current_expression_index]
    print(f"Current expression: {expression}. Press 'a' to start capturing images. Press 'w' to move to the next expression.")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):
            capture_mode = True

        if capture_mode and count < 50:
            img_path = os.path.join(base_dir, expression, f"{expression}_{count + 1}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
            print(f'Captured image {count} for: {expression}')

        if key == ord('w'):
            capture_mode = False
            current_expression_index += 1
            break

        if key == ord('q'):
            capture_mode = False
            current_expression_index = len(expressions)  # Exit loop
            break

    if count >= 50:
        print(f"Captured {count} images for: {expression}")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
