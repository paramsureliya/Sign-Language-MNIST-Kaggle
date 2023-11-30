import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('sign_language_cnn_model.keras')

# Define a mapping between class indices and alphabet letters
class_to_alphabet = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Open the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set the window size
cv2.namedWindow('Real-Time Sign Language Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-Time Sign Language Detection', 800, 600)  # Adjust the size as needed

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame (resize, convert to grayscale, etc.)
    # Example: resize the frame to 28x28 and convert to grayscale
    frame_resized = cv2.resize(frame, (28, 28))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray.reshape(1, 28, 28, 1)  # Adjust the shape to match your model's input shape

    # Make predictions using your model
    predictions = model.predict(frame_gray)
    predicted_class = np.argmax(predictions)
    predicted_alphabet = class_to_alphabet[predicted_class]

    # Display the frame with the predicted alphabet
    cv2.putText(frame, f"Predicted Alphabet: {predicted_alphabet}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Sign Language Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any windows
cap.release()
cv2.destroyAllWindows()