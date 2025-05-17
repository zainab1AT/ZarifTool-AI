import os
from xml.dom import minidom

out_dir = './outImages'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

file = minidom.parse("annotations/annotationsN2.xml")

images = file.getElementsByTagName('image')

for image in images:
    width = int(image.getAttribute('width'))
    height = int(image.getAttribute('height'))
    name = image.getAttribute('name')

    # Get bounding box information
    bbox = image.getElementsByTagName('box')[0]
    xtl = float(bbox.getAttribute('xtl'))
    ytl = float(bbox.getAttribute('ytl'))
    xbr = float(bbox.getAttribute('xbr'))
    ybr = float(bbox.getAttribute('ybr'))

    # Calculate the width and height of the bounding box
    w = xbr - xtl
    h = ybr - ytl

    # Open the label file for the YOLOv8 format
    with open(os.path.join(out_dir, name[:-4] + '.txt'), 'w') as label_file:

        # Normalize the bounding box
        center_x = (xtl + w / 2) / width
        center_y = (ytl + h / 2) / height
        norm_w = w / width
        norm_h = h / height

        # Write the bounding box in YOLO format (class_id, center_x, center_y, width, height)
        label_file.write(f"0 {center_x} {center_y} {norm_w} {norm_h} ")

        # Process points for the image (assumed to be keypoints)
        points_elem = image.getElementsByTagName('points')
        if points_elem:
            points = points_elem[0].getAttribute('points').split(';')

            # Normalize each point and add to the file
            for p in points:
                p1, p2 = map(float, p.split(','))
                norm_p1 = p1 / width
                norm_p2 = p2 / height
                label_file.write(f"{norm_p1} {norm_p2} 1 ")  # 1 represents visibility (you can adjust this)

        # Ensure the file ends with a newline
        label_file.write("\n")
