from ultralytics import YOLO
import cv2

model_path = 'C:/Users/Microsoft/runs/pose/train2/weights/last.pt'
image_path = 'data/images/train/k-side2.jpg'
img = cv2.imread(image_path)

model = YOLO(model_path)

# Perform inference
results = model(image_path)

# Extract keypoints and draw them on the image
for result in results:
    if hasattr(result, 'keypoints'):
        keypoints = result.keypoints.data.numpy()  # Convert to numpy array for easier handling
        for keypoint in keypoints[0]:  # We have one body with multiple keypoints
            x, y, conf = keypoint  # Extract the x, y, and confidence values
            if conf > 0.5:  # Only draw keypoints with a high confidence value
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw a red circle

# Display the image with keypoints
cv2.imshow('Keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
