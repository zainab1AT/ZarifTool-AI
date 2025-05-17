from ultralytics import YOLO
import cv2
import os

class PhysioModel:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.model = YOLO("model/best.pt")

        # Initialize keypoint coordinates
        self.first_ear_x = self.first_ear_y = None
        self.second_ear_x = self.second_ear_y = None
        self.neck_x = self.neck_y = None
        self.upper_back_x = self.upper_back_y = None
        self.lower_back_x = self.lower_back_y = None
        self.pelvic_back_x = self.pelvic_back_y = None
        self.pelvic_front_x = self.pelvic_front_y = None

        self.extract_keypoints()

    def extract_keypoints(self):
        results = self.model.predict(source=self.image, conf=0.3)
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            print("No detection or no keypoints.")
            return

        keypoints_tensor = results[0].keypoints.data
        if keypoints_tensor is None or len(keypoints_tensor) == 0:
            print("No keypoints data found.")
            return

        keypoints = keypoints_tensor[0].cpu().numpy()

        if keypoints is None or len(keypoints) < 7:
            print(f"Not enough keypoints detected. Only found {len(keypoints)} points.")
            return

        # Defensive unpacking
        try:
            points = []
            for i in range(7):
                x, y, _ = keypoints[i]
                if x is None or y is None:
                    points.append((None, None))
                else:
                    points.append((float(x), float(y)))

            (
                (self.first_ear_x, self.first_ear_y),
                (self.second_ear_x, self.second_ear_y),
                (self.neck_x, self.neck_y),
                (self.upper_back_x, self.upper_back_y),
                (self.lower_back_x, self.lower_back_y),
                (self.pelvic_back_x, self.pelvic_back_y),
                (self.pelvic_front_x, self.pelvic_front_y)
            ) = points

        except Exception as e:
            print(f"Error unpacking keypoints: {e}")



    def draw_keypoints(self, output_dir="physioOutImages", filename="output.jpg"):
        if self.first_ear_x is None:
            print("You must call extract_keypoints() first.")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)

        # Skeleton connections
        skeleton = [
            ((self.first_ear_x, self.first_ear_y), (self.neck_x, self.neck_y)),
            ((self.second_ear_x, self.second_ear_y), (self.neck_x, self.neck_y)),
            ((self.neck_x, self.neck_y), (self.upper_back_x, self.upper_back_y)),
            ((self.upper_back_x, self.upper_back_y), (self.lower_back_x, self.lower_back_y)),
            ((self.lower_back_x, self.lower_back_y), (self.pelvic_back_x, self.pelvic_back_y)),
            ((self.lower_back_x, self.lower_back_y), (self.pelvic_front_x, self.pelvic_front_y)),
        ]

        # Draw keypoints
        for (x, y) in [
            (self.first_ear_x, self.first_ear_y),
            (self.second_ear_x, self.second_ear_y),
            (self.neck_x, self.neck_y),
            (self.upper_back_x, self.upper_back_y),
            (self.lower_back_x, self.lower_back_y),
            (self.pelvic_back_x, self.pelvic_back_y),
            (self.pelvic_front_x, self.pelvic_front_y)
        ]:
            cv2.circle(self.image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Draw skeleton
        for (pt1, pt2) in skeleton:
            cv2.line(self.image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)

        cv2.imwrite(output_path, self.image)
        print(f"Saved image with keypoints to {output_path}")
