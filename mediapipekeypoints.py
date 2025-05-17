import cv2
import mediapipe as mp
import numpy as np
import math
from inference_sdk import InferenceHTTPClient
import base64

# Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

class PoseEstimator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError("Failed to load the image. Please check the file path or try another image.")
        
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.results = self.pose.process(self.image_rgb)

        
        if self.results.pose_landmarks:
            self.landmarks = self.results.pose_landmarks.landmark
            if not self.has_enough_keypoints():
                raise ValueError("Image is not good enough. Please try another one.")
            self.extract_keypoints()
        else:
            raise ValueError("No pose landmarks detected. Please use a clearer image.")
        
    def has_enough_keypoints(self):
        """ Check if the key landmarks have valid coordinates """
        required_landmarks = [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.LEFT_EYE
        ]

        for landmark in required_landmarks:
            x, y = self.get_coordinates(landmark)
            if x is None or y is None:
                return False  # Missing essential keypoints

        return True  # Enough keypoints to continue

    def extract_keypoints(self):
        self.right_shoulder_x, self.right_shoulder_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        self.left_shoulder_x, self.left_shoulder_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        
        self.right_elbow_x, self.right_elbow_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_ELBOW)
        self.left_elbow_x, self.left_elbow_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_ELBOW)
        
        self.right_wrist_x, self.right_wrist_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_WRIST)
        self.left_wrist_x, self.left_wrist_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_WRIST)
        
        self.right_hip_x, self.right_hip_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_HIP)
        self.left_hip_x, self.left_hip_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_HIP)
        
        self.right_knee_x, self.right_knee_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_KNEE)
        self.left_knee_x, self.left_knee_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_KNEE)
        
        self.right_ankle_x, self.right_ankle_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        self.left_ankle_x, self.left_ankle_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        
        self.right_eye_x, self.right_eye_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_EYE)
        self.left_eye_x, self.left_eye_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_EYE)
        
        self.right_ear_x, self.right_ear_y = self.get_coordinates(self.mp_pose.PoseLandmark.RIGHT_EAR)
        self.left_ear_x, self.left_ear_y = self.get_coordinates(self.mp_pose.PoseLandmark.LEFT_EAR)
    
    def get_coordinates(self, landmark):
        if self.landmarks:
            return self.landmarks[landmark].x, self.landmarks[landmark].y
        return None, None
    
    def is_front_view(self):
        shoulder_distance = abs(self.right_shoulder_x - self.left_shoulder_x)
        # hip_distance = abs(self.right_hip_x - self.left_hip_x)
        return shoulder_distance > 0.15 #and hip_distance > 0.15  # Adjust threshold as needed
    
    def draw_keypoints(self):
        if not self.landmarks:
            print("No pose landmarks detected.")
            return
        
        keypoints = {
            "Right Shoulder": (self.right_shoulder_x, self.right_shoulder_y),
            "Left Shoulder": (self.left_shoulder_x, self.left_shoulder_y),
            "Right Elbow": (self.right_elbow_x, self.right_elbow_y),
            "Left Elbow": (self.left_elbow_x, self.left_elbow_y),
            "Right Wrist": (self.right_wrist_x, self.right_wrist_y),
            "Left Wrist": (self.left_wrist_x, self.left_wrist_y),
            "Right Hip": (self.right_hip_x, self.right_hip_y),
            "Left Hip": (self.left_hip_x, self.left_hip_y),
            "Right Knee": (self.right_knee_x, self.right_knee_y),
            "Left Knee": (self.left_knee_x, self.left_knee_y),
            "Right Ankle": (self.right_ankle_x, self.right_ankle_y),
            "Left Ankle": (self.left_ankle_x, self.left_ankle_y),
            "Right Eye": (self.right_eye_x, self.right_eye_y),
            "Left Eye": (self.left_eye_x, self.left_eye_y),
            "Right Ear": (self.right_ear_x, self.right_ear_y),
            "Left Ear": (self.left_ear_x, self.left_ear_y)
        }
        
        for key, (x, y) in keypoints.items():
            if x and y:
                h, w, _ = self.image.shape
                cx, cy = int(x * w), int(y * h)
                cv2.circle(self.image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(self.image, key, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        display_width = 400  # Adjust this as needed
        h, w, _ = self.image.shape
        scaling_factor = display_width / w
        new_height = int(h * scaling_factor)
        resized_image = cv2.resize(self.image, (display_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow("Pose Landmarks", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __del__(self):
        """Release the Mediapipe Pose model when the object is deleted."""
        if hasattr(self, 'pose'):
            self.pose.close()

    
    def uneven_shoulders(self):
        shoulder_y_distance = abs(self.right_shoulder_y - self.left_shoulder_y)
        shoulder_y_distance_on = (self.right_shoulder_y / self.left_shoulder_y) > 2
        shoulder_y_distance_on_op = (self.left_shoulder_y / self.right_shoulder_y) > 2
        return shoulder_y_distance > 0.007
    
    def uneven_shoulders_right_higher(self):
        if self.uneven_shoulders():
            right_is_higher = self.right_shoulder_y - self.left_shoulder_y
            if right_is_higher > 0:
                return True
        return False
    
    def uneven_shoulders_left_higher(self):
        if self.uneven_shoulders():
            left_is_higher = self.left_shoulder_y - self.right_shoulder_y
            if left_is_higher > 0:
                return True
        return False
    
    def uneven_hips(self):
        hips_y_distance = abs(self.right_hip_y - self.left_hip_y)
        knees_y_distance = abs(self.right_knee_y - self.left_knee_y)
        return (hips_y_distance > 0.005 or knees_y_distance > 0.0045) #and ((hips_right_higher) or (hips_left_higher))
    
    def knee_right(self):
        return (self.right_knee_y - self.left_knee_y)>0
    def hip_right(self):
        return (self.right_hip_y - self.left_hip_y)>0
    
    def uneven_hips_right_higher(self):
        if self.uneven_hips():
            right_knee_higher = (self.right_knee_y - self.left_knee_y) > 0
            right_is_higher = self.right_hip_y - self.left_hip_y
            hips_right_higher = right_is_higher and right_knee_higher
            if (right_is_higher > 0) or (hips_right_higher):
                return True
        return False
    
    def uneven_hips_left_higher(self):
        if self.uneven_hips():
            left_knee_higher = (self.left_knee_y - self.right_knee_y) > 0
            left_is_higher = self.left_hip_y - self.right_hip_y
            hips_left_higher = left_is_higher and left_knee_higher
            if (left_is_higher > 0) or (hips_left_higher):
                return True
        return False
    
    def scoliosis(self):
        if (self.uneven_shoulders_right_higher() and self.uneven_hips_left_higher()) or (self.uneven_shoulders_left_higher() and self.uneven_hips_right_higher()):
            return True
        return False
    
    def knees_distance(self):
        return abs(self.right_knee_x - self.left_knee_x)
    
    def ankels_distance(self):
        return abs(self.right_ankle_x - self.left_ankle_x)
    
    def hips_distance(self):
        return abs(self.right_hip_x - self.left_hip_x)
    
    def detect_knock_knees(self):
        return abs(self.hips_distance() / self.knees_distance()) > 0.8 and abs(self.knees_distance() / self.ankels_distance()) < 0.8
    
    def detect_bow_knees(self):
        return (self.knees_distance() > self.ankels_distance()) and (self.knees_distance() > self.hips_distance()) and ((self.knees_distance()-self.ankels_distance())>0.025)

    def detect_knee_hyperextension_right(self):
        if self.detect_side_view_orientation() == "Left Side":
            if self.right_knee_x > self.right_ankle_x:
                return True
        elif self.detect_side_view_orientation() == "Right Side":
            if self.right_knee_x < self.right_ankle_x:
                return True
        return False
    
    def detect_knee_hyperextension_left(self):
        if self.detect_side_view_orientation() == "Left Side":
            if self.left_knee_x > self.left_ankle_x:
                return True
        elif self.detect_side_view_orientation() == "Right Side":
            if self.left_knee_x < self.left_ankle_x:
                return True   
        return False
    
    def detect_side_view_orientation(self):
   
        if not self.is_front_view():
            
            if (self.right_ear_x > self.right_eye_x) and (self.left_ear_x > self.left_eye_x):
                return "Left Side"
            else:
                return "Right Side"
        
        return "Unknown"
    
    #until our model is ready
    def rounded_shoulders(self):
        if not self.is_front_view():
            if self.detect_side_view_orientation() == "Left Side":
                if self.left_shoulder_x < self.left_hip_x:
                    return abs(self.left_shoulder_x - self.left_hip_x)
            elif self.detect_side_view_orientation() == "Right Side":
                if self.left_shoulder_x > self.left_hip_x:
                    return abs(self.left_shoulder_x - self.left_hip_x)
        return False
    
    #until our model is ready
    def forward_head(self):
        if self.detect_side_view_orientation() == "Left Side":
            return abs(self.left_eye_x - self.left_shoulder_x) > 0.21
        elif self.detect_side_view_orientation() == "Right Side":
            return abs(self.right_eye_x - self.right_hip_x) > 0.21
        return False
    
    def detect_all_problems_from_front(self):
        front_problems = []
        if self.is_front_view():
            if self.uneven_shoulders_left_higher():
                front_problems.append("Uneven Shoulders: Left Shoulder Higher")
            if self.uneven_shoulders_right_higher():
                front_problems.append("Uneven Shoulders: Right Shoulder Higher")
            if self.uneven_hips_right_higher():
                front_problems.append("Uneven Hips: Right Hip Higher")
            if self.uneven_hips_left_higher():
                front_problems.append("Uneven Hips: Left Hip Higher")
            if self.detect_bow_knees():
                front_problems.append("Bow Knees")
            if self.detect_knock_knees():
                front_problems.append("Knock Knees")
            if self.scoliosis():
                front_problems.append("Scoliosis")
            return front_problems
        return ["side view"]
    
    def detect_all_problems_from_side(self):
        side_problems = []
        if not self.is_front_view():
            if self.forward_head():
                side_problems.append("Forward Head")
            if self.rounded_shoulders():
                side_problems.append("Rounded Shoulders")
            if self.detect_knee_hyperextension_left():
                side_problems.append("Knee Hyperextension: Left")
            if self.detect_knee_hyperextension_right():
                side_problems.append("Knee Hyperextension: Right")
            return side_problems
        return ["side view"]
    