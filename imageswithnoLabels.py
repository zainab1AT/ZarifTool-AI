import os
import shutil

# Path to your dataset (update this to match your setup)
dataset_path = "C:/projectstask_physio/data"

output_folder = os.path.join(dataset_path, "./images_no_label")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def check_and_move_unlabeled_images(split):
    images_path = os.path.join(dataset_path, "images", split)
    labels_path = os.path.join(dataset_path, "labels", split)

    missing_labels = []

    for image in os.listdir(images_path):
        if image.endswith((".jpg", ".png", ".jpeg")):  # Check only image files
            label_file = os.path.splitext(image)[0] + ".txt"
            label_path = os.path.join(labels_path, label_file)

            if not os.path.exists(label_path):  # If label file is missing
                missing_labels.append(image)
                shutil.move(os.path.join(images_path, image), os.path.join(output_folder, image))

    if missing_labels:
        print(f"❌ Moved {len(missing_labels)} images with missing labels to '{output_folder}':")
        for img in missing_labels:
            print(f"   - {img}")
    else:
        print(f"✅ All images in '{split}' have corresponding labels.")

# Check both training and validation sets
check_and_move_unlabeled_images("train")
check_and_move_unlabeled_images("val")

