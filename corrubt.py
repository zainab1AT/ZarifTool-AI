from PIL import Image
import os

# Path to your images directory
images_dir = "C:/Users/Microsoft/Downloads/data-20250209T153416Z-001/data/images_no_label"
#"C:/Users/Microsoft/Downloads/data-20250209T153416Z-001/data/images/train"

# List to store corrupt files
corrupt_images = []

for img_name in os.listdir(images_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        try:
            with Image.open(os.path.join(images_dir, img_name)) as img:
                img.verify()  # Verify the integrity of the image
        except (IOError, SyntaxError) as e:
            corrupt_images.append(img_name)

if corrupt_images:
    print("Corrupt image files found:")
    for corrupt_file in corrupt_images:
        print(corrupt_file)
else:
    print("No corrupt image files found.")