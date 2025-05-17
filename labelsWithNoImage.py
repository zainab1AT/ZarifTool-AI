import os
import shutil

# Path to dataset (update if needed)
dataset_path = "C:/projectstask_physio/data"
labels_with_no_images_folder = os.path.join(dataset_path, "labels_with_no_images")

# Create the "labels_with_no_images" folder if it doesn't exist
os.makedirs(labels_with_no_images_folder, exist_ok=True)

def check_unmatched_labels(split):
    images_path = os.path.join(dataset_path, "images", split)
    labels_path = os.path.join(dataset_path, "labels", split)

    missing_images = []

    for label in os.listdir(labels_path):
        if label.endswith(".txt"):  # Only check label files
            image_jpg = os.path.splitext(label)[0] + ".jpg"
            image_png = os.path.splitext(label)[0] + ".png"
            image_jpeg = os.path.splitext(label)[0] + ".jpeg"

            # Check if any corresponding image exists
            if not any(os.path.exists(os.path.join(images_path, img)) for img in [image_jpg, image_png, image_jpeg]):
                missing_images.append(label)

                # Move the label to "labels_with_no_images" folder
                label_path = os.path.join(labels_path, label)
                shutil.move(label_path, os.path.join(labels_with_no_images_folder, label))

    if missing_images:
        print(f"❌ Found {len(missing_images)} labels without images in '{split}':")
        for lbl in missing_images:
            print(f"   - {lbl}")
    else:
        print(f"✅ All labels in '{split}' have corresponding images.")

# Check both training and validation sets
check_unmatched_labels("train")
check_unmatched_labels("val")
