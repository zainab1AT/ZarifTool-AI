# import cv2
# import numpy as np
# import requests
# from mediapipekeypoints import PoseEstimator

# def download_and_save_image(image_url, filename="temp_image.jpg"):
#     """Download an image from a URL and save it locally."""
#     response = requests.get(image_url)  # Fetch the image from URL
#     if response.status_code == 200:
#         # Convert image content into numpy array
#         nparr = np.frombuffer(response.content, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         cv2.imwrite(filename, img)  # Save the image
#         return filename  # Return the file path
#     else:
#         raise Exception(f"Failed to download image: {response.status_code}")

# def resize_image(image_path, target_width=None, target_height=None):
#     """
#     Resize an image while preserving its aspect ratio.
#     Specify either target_width or target_height.
#     """
#     image_np = cv2.imread(image_path)
#     original_height, original_width = image_np.shape[:2]

#     # Calculate the scaling factor
#     if target_width is not None:
#         scaling_factor = target_width / original_width
#         new_width = target_width
#         new_height = int(original_height * scaling_factor)
#     elif target_height is not None:
#         scaling_factor = target_height / original_height
#         new_height = target_height
#         new_width = int(original_width * scaling_factor)
#     else:
#         raise ValueError("Either target_width or target_height must be specified.")

#     # Resize the image
#     resized_image = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)

#     # Save the resized image
#     cv2.imwrite(image_path, resized_image)
#     return image_path

# # Download and save image
# image_url = "http://res.cloudinary.com/dvp4ilk19/image/upload/v1741875940/08a91d21-9d99-4b81-aac3-f081648eac98.jpg"
# local_image_path = download_and_save_image(image_url, "images/temp_image.jpg")

# # Resize the image to a target width of 800 pixels (height will be scaled proportionally)
# local_image_path = resize_image(local_image_path, target_width=800)

# # Display the downloaded image
# image_np = cv2.imread(local_image_path)
# # cv2.imshow("Resized Image", image_np)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# print(f"Resized image saved to: {local_image_path}")

# # Initialize PoseEstimator
# pose_estimator = PoseEstimator("resized_image.jpg")

# # Print keypoints for debugging"C:/Users/Microsoft/Downloads/AD515A6E-4E0E-4B3E-B1DD-B20CA4D4B995.jpeg"
# # print("Keypoints:", pose_estimator.keypoints)
# print(abs(pose_estimator.right_shoulder_y - pose_estimator.left_shoulder_y))
# print(abs(pose_estimator.right_hip_y - pose_estimator.left_hip_y))
# print(abs(pose_estimator.right_kanee_y - pose_estimator.left_knee_y))

# # 0.014940828084945679
# # 0.001989811658859253
# # 0.007894635200500488
# # Detect problems
# if pose_estimator.is_front_view():
#     print("Front View Problems:", pose_estimator.detect_all_problems_from_front())
# else:
#     print("Side View Problems:", pose_estimator.detect_all_problems_from_side())

# # Optionally, draw the keypoints
# pose_estimator.draw_keypoints()

# del pose_estimator  # This will trigger the __del__ method

from TorchDetector import PoseDetector
from mediapipekeypoints import PoseEstimator
from physiomodel import PhysioModel

"""
normal
(female)
(287) lordosis 1.0765555




normal 
(male)
(114) lordosis 1.0996232
(231) lordosis 1.0764213

not
(male)
(278) lordosis 1.0604004
(283) lordosis 1.0979719
(311) lordosis 1.0764213
"""
"""
kyphosis
13
65
73
108
279
300
372

kind of
(nour) lordosis distacne 0.8592417

kyphosis
(13) Kyphosis  1.1526759
(65) Kyphosis  1.1326604
(108) Kyphosis  1.1375958
(300) Kyphosis  1.2043196
(372) Kyphosis  1.2500441


normal
(73) Kyphosis  0.9927582
(279) Kyphosis  1.0843434
(1) Kyphosis  0.976142
(2) Kyphosis  0.9697683
(3) Kyphosis  1.0028892
(4) Kyphosis  0.9594563

"""
number = 4
image_path = f"images/temp_image.jpg"
#"D:/A Graduation Project/wrong images/Images Taken JPG-20250417T201409Z-001/Images Taken JPG/imagesTakenC({number}).jpg"
#"C:/Users/Microsoft/Downloads/mo ({number}).jpeg"
#"C:/Users/Microsoft/Downloads/me-image (2).jpeg"
#"C:/Users/Microsoft/Downloads/sb (8).jpeg"
#"C:/Users/Microsoft/Downloads/lo.jpeg"
#"C:/Users/Microsoft/Downloads/WhatsApp Image 2025-04-22 at 8.45.49 PM.jpeg"
#"D:/A Graduation Project/test images/me-test (2).jpeg"
#"D:/A Graduation Project/wrong images/Images Taken JPG-20250417T201409Z-001/Images Taken JPG/imagesTakenC(185).jpg"
#"C:/Users/Microsoft/Downloads/testt (3).jpeg"
#"D:/A Graduation Project/collected images/a9718c26-22ed-41c6-9a03-2837dc7fe10f.png"
#"physioOutImages/normal233_1.07.jpg"
# "D:/A Graduation Project/wrong images/Images Taken JPG-20250417T201409Z-001/Images Taken JPG/imagesTakenC(233).jpg"
#"C:/Users/Microsoft/Downloads/testt (2).jpeg"
#"D:/A Graduation Project/test images/t.png"
# "C:/Users/Microsoft/Downloads/Exaggerated Posture in Neutral Light.png"

physio = PhysioModel(image_path)
posture_detector = PoseDetector(image_path)
pose_estimator = PoseEstimator(image_path) 

physio.extract_keypoints()
side_view = False if posture_detector.is_front_view() else True
rounded_shoulders = None
fb = None

if (posture_detector.lordosis1(physio.lower_back_x, physio.neck_x)):
    rounded_shoulders = True
else:
    if posture_detector.detect_side_view_orientation() == "Right Side":
        rounded_shoulders = physio.lower_back_x / physio.neck_x
    else:
        rounded_shoulders = physio.neck_x / physio.lower_back_x

if posture_detector.detect_side_view_orientation() == "Right Side":
    fb = posture_detector.right_ear_x / physio.neck_x
else:
    fb = physio.neck_x / posture_detector.left_ear_x


print("is side view:", side_view)
print("the body is facing:", posture_detector.detect_side_view_orientation())
print("lordosis distacne", rounded_shoulders)
print("sway back", posture_detector.sway_back())
print("lordosis", posture_detector.lordosis(physio.lower_back_x, physio.neck_x))
print("forward head?", posture_detector.forward_head(physio.neck_x))
print("Kyphosis ", posture_detector.kyphosis())
problems = posture_detector.detect_all_problems_from_side(pose_estimator.detect_knee_hyperextension_left(), pose_estimator.detect_knee_hyperextension_right(), physio.lower_back_x, physio.neck_x)
print("Detected problems:", problems)

filename = f"me{number}_{posture_detector.kyphosis():.2f}.jpg"
physio.draw_keypoints(output_dir="physioOutImages", filename=filename)
posture_detector.draw_keypoints()