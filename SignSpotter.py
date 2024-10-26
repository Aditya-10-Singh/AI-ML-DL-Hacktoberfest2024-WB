import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained MobileNetV2 model for feature extraction
model = MobileNetV2(weights="imagenet")

# Labels for traffic signs (Note: This example only includes a few categories for illustration)
traffic_sign_labels = {
    920: "Stop sign",           # ImageNet class for stop sign
    919: "Traffic light",        # ImageNet class for traffic light
    918: "Parking meter",        # ImageNet class for parking meter (placeholder)
    # Add more traffic sign labels as needed
}

def detect_traffic_sign(frame):
    # Resize the frame to match the input shape of MobileNetV2
    resized_frame = cv2.resize(frame, (224, 224))
    image_array = img_to_array(resized_frame)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    # Run inference
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    # Check for traffic signs in the top predictions
    for (imagenet_id, label, score) in decoded_predictions:
        if imagenet_id in traffic_sign_labels:
            detected_label = traffic_sign_labels[imagenet_id]
            confidence = f"{score * 100:.2f}%"
            return detected_label, confidence
    return None, None

# Start video capture
cap = cv2.VideoCapture(0)  # Use '0' for webcam or replace with video path

print("Starting video stream...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect traffic sign in the current frame
    label, confidence = detect_traffic_sign(frame)
    
    # Display the detection result on the frame
    if label is not None:
        text = f"{label}: {confidence}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (5, 5), (235, 40), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Traffic Sign Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
