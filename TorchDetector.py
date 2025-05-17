import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from mediapipekeypoints import PoseEstimator
from physiomodel import PhysioModel

class PoseDetector: 
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path).convert("RGB")
        self.physio = PhysioModel(image_path)

        self.nose_x = self.nose_y = None
        self.left_eye_x = self.left_eye_y = None
        self.right_eye_x = self.right_eye_y = None
        self.left_ear_x = self.left_ear_y = None
        self.right_ear_x = self.right_ear_y = None
        self.left_shoulder_x = self.left_shoulder_y = None
        self.right_shoulder_x = self.right_shoulder_y = None
        self.left_elbow_x = self.left_elbow_y = None
        self.right_elbow_x = self.right_elbow_y = None
        self.left_wrist_x = self.left_wrist_y = None
        self.right_wrist_x = self.right_wrist_y = None
        self.left_hip_x = self.left_hip_y = None
        self.right_hip_x = self.right_hip_y = None
        self.left_knee_x = self.left_knee_y = None
        self.right_knee_x = self.right_knee_y = None
        self.left_ankle_x = self.left_ankle_y = None
        self.right_ankle_x = self.right_ankle_y = None
        
        # Load model and process image
        self.model = keypointrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.image_tensor = self.preprocess_image(self.image)
        
        with torch.no_grad():
            output = self.model(self.image_tensor)

        self.keypoints = output[0]['keypoints'][0].numpy()
        self.scores = output[0]['scores'][0].numpy()

        if not self.has_enough_keypoints():
            raise ValueError("Image is not good enough. Please try another one.")

        self.extract_keypoints()
    
    def extract_keypoints(self):
        """Extracts keypoints and assigns values to explicitly defined attributes."""
        coco_keypoints = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        self.landmarks = {}
        for i, (x, y, confidence) in enumerate(self.keypoints):
            if confidence > 0.5:
                key_name = coco_keypoints[i]
                self.landmarks[key_name] = (x, y)

                # Assign known keypoints to explicitly defined attributes
                if key_name == "nose":
                    self.nose_x, self.nose_y = x, y
                elif key_name == "left_eye":
                    self.left_eye_x, self.left_eye_y = x, y
                elif key_name == "right_eye":
                    self.right_eye_x, self.right_eye_y = x, y
                elif key_name == "left_ear":
                    self.left_ear_x, self.left_ear_y = x, y
                elif key_name == "right_ear":
                    self.right_ear_x, self.right_ear_y = x, y
                elif key_name == "left_shoulder":
                    self.left_shoulder_x, self.left_shoulder_y = x, y
                elif key_name == "right_shoulder":
                    self.right_shoulder_x, self.right_shoulder_y = x, y
                elif key_name == "left_elbow":
                    self.left_elbow_x, self.left_elbow_y = x, y
                elif key_name == "right_elbow":
                    self.right_elbow_x, self.right_elbow_y = x, y
                elif key_name == "left_wrist":
                    self.left_wrist_x, self.left_wrist_y = x, y
                elif key_name == "right_wrist":
                    self.right_wrist_x, self.right_wrist_y = x, y
                elif key_name == "left_hip":
                    self.left_hip_x, self.left_hip_y = x, y
                elif key_name == "right_hip":
                    self.right_hip_x, self.right_hip_y = x, y
                elif key_name == "left_knee":
                    self.left_knee_x, self.left_knee_y = x, y
                elif key_name == "right_knee":
                    self.right_knee_x, self.right_knee_y = x, y
                elif key_name == "left_ankle":
                    self.left_ankle_x, self.left_ankle_y = x, y
                elif key_name == "right_ankle":
                    self.right_ankle_x, self.right_ankle_y = x, y

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)

    def has_enough_keypoints(self):
        """Checks if the image has a sufficient number of keypoints with confidence > 0.5"""
        return sum(1 for kp in self.keypoints if kp[2] > 0.5) >= 5  # Adjust threshold as needed

    def draw_keypoints(self):
        """Draws keypoints and their names along with (x, y) coordinates on the image."""
        img = np.array(self.image)  # Convert PIL image to NumPy array

        # Resize the image to fit within a smaller window
        display_width = 400  # Adjust as needed
        h, w = img.shape[:2]
        scaling_factor = display_width / w
        new_height = int(h * scaling_factor)
        img = cv2.resize(img, (display_width, new_height), interpolation=cv2.INTER_AREA)

        for key_name, (x, y) in self.landmarks.items():
            scaled_x, scaled_y = int(x * scaling_factor), int(y * scaling_factor)  # Scale keypoints
            cv2.circle(img, (scaled_x, scaled_y), 5, (0, 255, 0), -1)

            # Display keypoint name along with (x, y) coordinates in purple and smaller font
            text = f"{key_name} ({int(x)}, {int(y)})"
            cv2.putText(img, text, (scaled_x + 5, scaled_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 0, 128), 1)  # Purple color (BGR: 128, 0, 128)

        cv2.imshow("Pose Keypoints", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    def is_front_view (self):
        """Checks if the image is a front view of the person"""
        x_ear_distance = self.left_ear_x / self.right_ear_x
        threshold = 1.1
        return x_ear_distance > 1.1
    
    def uneven_shoulders_left_higher (self, uneven):
        """Checks if the shoulders are uneven"""
        if self.uneven_shoulders(uneven):
            return self.left_shoulder_y > self.right_shoulder_y
        return False
    
    def uneven_shoulders_right_higher (self, uneven):
        """Checks if the shoulders are uneven"""
        if self.uneven_shoulders(uneven):
            return self.right_shoulder_y > self.left_shoulder_y
        return False  
    
    def shoulders_slope (self):
        """Calculates the slope of the shoulders"""
        return (self.right_shoulder_y - self.left_shoulder_y) / (self.right_shoulder_x - self.left_shoulder_x)
    
    def uneven_shoulders(self, uneven):
        """Classifies if shoulders are uneven based on slope and distance ratios"""
        threshold = 0.043
        slope = self.shoulders_slope()
        return (abs(slope) > threshold) or uneven
    
    def hip_slope (self):
        """Calculates the slope of the hips"""
        return (self.right_hip_y - self.left_hip_y) / (self.right_hip_x - self.left_hip_x)
    
    def uneven_hips (self):
        """Classifies if hips are uneven based on slope and distance ratios"""
        threshold = 0.06
        slope = self.hip_slope()
        return abs(slope) > threshold
    
    def uneven_hips_left_higher (self):
        if self.uneven_hips():
            return self.left_hip_y > self.right_hip_y
        return False
    
    def uneven_hips_right_higher (self):
        if self.uneven_hips():
            return self.right_hip_y > self.left_hip_y
        return False
    
    def scoliosis (self, uneven):
        if (self.uneven_shoulders_right_higher(uneven) and self.uneven_hips_left_higher()) or (self.uneven_shoulders_left_higher(uneven) and self.uneven_hips_right_higher()):
            return True
        return False
    
    def knees_distance(self):
        return self.right_knee_x / self.left_knee_x
    
    def ankels_distance(self):
        return self.right_ankle_x / self.left_ankle_x
    
    def hips_distance(self):
        return self.right_hip_x / self.left_hip_x
    
    def detect_knee_hyperextension_right(self, hke_pose_right):
        threshold = 0.9
        if self.detect_side_view_orientation() == "Left Side":
            if ((self.right_knee_x > self.right_ankle_x) and (self.right_knee_x > self.right_hip_x) and ((self.right_ankle_x / self.right_knee_x)<threshold)) or hke_pose_right:
                return True
        elif self.detect_side_view_orientation() == "Right Side":
            if ((self.right_knee_x < self.right_ankle_x) and (self.right_knee_x < self.right_hip_x) and ((self.right_knee_x / self.right_ankle_x)<threshold)) or hke_pose_right:
                return True
        return False
    
    def detect_knee_hyperextension_left(self, hke_pose_left):
        threshold = 0.9
        if self.detect_side_view_orientation() == "Left Side":
            if ((self.left_knee_x > self.left_ankle_x) and (self.left_knee_x > self.left_hip_x) and ((self.left_ankle_x / self.left_knee_x)<threshold)) or hke_pose_left:
                return True
        elif self.detect_side_view_orientation() == "Right Side":
            if ((self.left_knee_x < self.left_ankle_x) and (self.left_knee_x < self.left_hip_x) and ((self.left_knee_x / self.left_ankle_x)<threshold)) or hke_pose_left:
                return True   
        return False
    
    def detect_side_view_orientation(self):
   
        if not self.is_front_view():
            
            if (self.right_ear_x > self.right_eye_x) and (self.left_ear_x > self.left_eye_x):
                return "Left Side"
            else:
                return "Right Side"
        
        return "Unknown"
    
    #until our model is ready **************************
    def rounded_shoulders(self):
        if not self.is_front_view():
            if self.detect_side_view_orientation() == "Left Side":
                if self.left_shoulder_x < self.left_hip_x:
                    return (self.left_shoulder_x / self.left_hip_x)
            elif self.detect_side_view_orientation() == "Right Side":
                if self.left_shoulder_x > self.left_hip_x:
                    return (self.left_shoulder_x / self.left_hip_x)
        return False
    
    def forward_head (self, neck_x):
        forward_neck = None
        forward_shoulder= None

        if self.detect_side_view_orientation() == "Right Side":
            forward_neck = self.right_ear_x / neck_x
        else:
            forward_neck = neck_x / self.left_ear_x
        
        if self.detect_side_view_orientation() == "Right Side":
            forward_shoulder = self.right_ear_x / self.right_shoulder_x
        else:
            forward_shoulder = self.left_shoulder_x / self.left_ear_x
        
        return (forward_neck > 1.1) and (forward_shoulder > 1.058)
    
    def lordosis1 (self, lower_back, neck):
        if self.detect_side_view_orientation() == "Right Side":
            return  neck < lower_back
        return lower_back < neck
    
    def lordosis (self, lower_back, neck):
        if (self.lordosis1(lower_back, neck)):
            return True
        else:
            if self.detect_side_view_orientation() == "Right Side":
                return (lower_back / neck) > 0.95
            else:
                return (neck / lower_back) > 0.95
    
    def anterior_pelvic_tilt (self, lower_back, neck):
        if (self.lordosis(lower_back, neck)):
            return True
        else:
            return False
    
    def sway_back (self):
        if self.detect_side_view_orientation() == "Right Side":
            return  (self.right_hip_x / self.right_shoulder_x) > 1.15
        return (self.left_shoulder_x/ self.right_hip_x) > 1.15
    
    def kyphosis (self):
        if self.detect_side_view_orientation() == "Right Side":
            return (self.right_shoulder_x / self.right_hip_x)>1.1
        return (self.left_hip_x / self.left_shoulder_x)>1.1
    
    def detect_all_problems_from_front(self, uneven_shoulders, detect_bow_knees, detect_knock_knees):
        front_problems = []
        if self.is_front_view():
            if self.uneven_shoulders_left_higher(uneven_shoulders):
                front_problems.append("Uneven Shoulders: Left Shoulder Higher")
            if self.uneven_shoulders_right_higher(uneven_shoulders):
                front_problems.append("Uneven Shoulders: Right Shoulder Higher")
            if self.uneven_hips_right_higher():
                front_problems.append("Uneven Hips: Right Hip Higher")
            if self.uneven_hips_left_higher():
                front_problems.append("Uneven Hips: Left Hip Higher")
            if detect_bow_knees:
                front_problems.append("Bow Knees")
            if detect_knock_knees:
                front_problems.append("Knock Knees")
            if self.scoliosis(uneven_shoulders):
                front_problems.append("Scoliosis")
            return front_problems
        return ["side view"]
    
    def detect_all_problems_from_side(self, detect_knee_hyperextension_left, detect_knee_hyperextension_right, lower_back, neck):
        side_problems = []
        if not self.is_front_view():
            if self.forward_head(neck):
                side_problems.append("Forward Head Posture")
            # if self.rounded_shoulders():
            #     side_problems.append("Rounded Shoulders")
            if self.detect_knee_hyperextension_left(detect_knee_hyperextension_left):
                side_problems.append("Knee Hyperextension: Left")
            if self.detect_knee_hyperextension_right(detect_knee_hyperextension_right):
                side_problems.append("Knee Hyperextension: Right")
            if self.sway_back():
                side_problems.append("Sway Back")
            if self.lordosis(lower_back, neck):
                side_problems.append("Lordosis")
            if self.anterior_pelvic_tilt(lower_back, neck):
                side_problems.append("Anterior Pelvic Tilt")
            if self.kyphosis():
                side_problems.append("Kyphosis")
            return side_problems
        return ["side view"]


# lordosis
# rounded shoulders
# anterior posteriori p t
# flat back
# kyphosis