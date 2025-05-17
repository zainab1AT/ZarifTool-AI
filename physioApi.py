import requests
import cv2
import os
from TorchDetector import PoseDetector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mediapipekeypoints import PoseEstimator
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
import base64

from physiomodel import PhysioModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_uri: str
    scan_type: str

def download_and_save_image(image_url, filename="temp_image.jpg"):
    """Download an image from a URL and save it locally."""
    response = requests.get(image_url, stream=True)

    if response.status_code != 200:
        raise ValueError(f"Failed to download image. HTTP status: {response.status_code}")

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise ValueError(f"URL does not contain a valid image. Content-Type: {content_type}")

    with open(filename, "wb") as file:
        file.write(response.content)

    return filename  # Return the file path

def xray_scoliosis(image_path):
    """Process X-ray image for scoliosis detection."""
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="YWIK2Hl85sFpzNLlEvt6"
    )

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    result = CLIENT.infer(encoded_image, model_id="scoliosis2-dvnfp-slhfq-x8hr5-qingn/4")
    
    scoliosis_detected = any(
        prediction["confidence"] > 0.8 for prediction in result.get("predictions", [])
    )

    return scoliosis_detected

@app.post("/detect-problems")
def create_item(request: ImageRequest):
    try:
        print("📷 Received Image URI:", request.image_uri)
        print("🛠️ Scan Type:", request.scan_type)

        if not request.image_uri:
            raise HTTPException(status_code=400, detail="Image URI is missing")

        # Force HTTPS
        secure_url = request.image_uri.replace("http://", "https://")

        # Step 1: Download & Save Image Locally
        local_image_path = download_and_save_image(secure_url, "images/temp_image.jpg")
        
        # Ensure the file exists before processing
        if not os.path.exists(local_image_path):
            raise ValueError(f"Image file does not exist at path: {local_image_path}")

        print("Local image path:", local_image_path)  # Debug print

        # Step 2: Read Image using OpenCV
        image_np = cv2.imread(local_image_path)

        if image_np is None:
            raise ValueError("Failed to read saved image. File may be corrupted.")

        # Step 3: Process Image with Pose Estimator
        # Inside create_item function
        if request.scan_type == "postureScan":
            pose_estimator = PoseEstimator(local_image_path)
            posture_detector = PoseDetector(local_image_path)
            physio_model = PhysioModel(local_image_path)

            # Debugging logs
            print(f"Lower back X: {physio_model.lower_back_x}, Neck X: {physio_model.neck_x}")

            # Check if keypoints are extracted properly
            if physio_model.lower_back_x is None or physio_model.neck_x is None:
                raise HTTPException(status_code=400, detail="Missing keypoints for posture analysis.")

            # Check for front or side view
            if posture_detector.is_front_view():
                problems = posture_detector.detect_all_problems_from_front(
                    pose_estimator.uneven_shoulders(),
                    pose_estimator.detect_bow_knees(),
                    pose_estimator.detect_knock_knees()
                )
            else:  # For side view
                problems = posture_detector.detect_all_problems_from_side(
                    pose_estimator.detect_knee_hyperextension_left(),
                    pose_estimator.detect_knee_hyperextension_right(),
                    physio_model.lower_back_x,
                    physio_model.neck_x
                )

            # Cleanup: Delete temp image file after processing
            os.remove(local_image_path)

            return {"problems": problems}


        elif request.scan_type == "xray":
            scoliosis_detected = xray_scoliosis(local_image_path)
            if scoliosis_detected:
                return {"problems": ["Scoliosis"]}
            else:
                return {"problems": []}

    except ValueError as e:
        print("❌ ValueError:", e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print("🚨 Unexpected Error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
