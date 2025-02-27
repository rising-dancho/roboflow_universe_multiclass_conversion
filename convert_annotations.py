import os
import cv2
import numpy as np
from tqdm import tqdm

# Converts XMin, YMin, XMax, YMax to normalized YOLO format
def convert(filename_str, coords):
    image_path = os.path.join(
        ROOT_DIR, "OID", "Dataset", DIR, CLASS_DIR, f"{filename_str}.jpg"
    )
    # print(f"Looking for image: {image_path}")

    if not os.path.exists(image_path):
        print(f"⚠️ Warning: File not found -> {image_path}")
        return None  # Return None to indicate failure

    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Warning: Failed to load -> {image_path}")
        return None  # Return None to indicate failure

    # Convert from (XMin, YMin, XMax, YMax) to (center_x, center_y, width, height)
    width = coords[2] - coords[0]
    height = coords[3] - coords[1]
    center_x = coords[0] + width / 2
    center_y = coords[1] + height / 2

    # Normalize values
    center_x /= image.shape[1]
    center_y /= image.shape[0]
    width /= image.shape[1]
    height /= image.shape[0]

    return [center_x, center_y, width, height]

ROOT_DIR = os.getcwd()

# Create dict to map class names to numbers for YOLO
classes = {}
with open("classes.txt", "r") as myFile:
    for num, line in enumerate(myFile, 0):
        classes[line.strip()] = num

# Move into dataset directory
dataset_path = os.path.join(ROOT_DIR, "OID", "Dataset")
DIRS = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Define class mapping
class_mapping = {"Hammer": 0, "Pipes": 1}

# Process all train, validation, and test folders
for DIR in DIRS:
    dir_path = os.path.join(dataset_path, DIR)
    print(f"Currently in subdirectory: {DIR}")

    CLASS_DIRS = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    for CLASS_DIR in CLASS_DIRS:
        class_dir_path = os.path.join(dir_path, CLASS_DIR)
        label_dir = os.path.join(class_dir_path, "Label")

        if not os.path.exists(label_dir):
            continue  # Skip if Label folder does not exist

        print(f"Converting annotations for class: {CLASS_DIR}")

        for filename in tqdm(os.listdir(label_dir)):
            if filename.endswith(".txt"):
                filename_str = os.path.splitext(filename)[0]  # Get filename without extension
                annotations = []

                file_path = os.path.join(label_dir, filename)

                with open(file_path, "r") as f:
                    for line in f:
                        labels = line.split()

                        if len(labels) < 5:
                            print(f"⚠️ Skipping invalid annotation in {filename}")
                            continue

                        # Set class ID based on folder name
                        class_id = class_mapping.get(CLASS_DIR, 0)

                        # Extract only (XMin, YMin, XMax, YMax)
                        coords = np.asarray([float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])])
                        converted_coords = convert(filename_str, coords)

                        if converted_coords is None:
                            continue  # Skip if conversion failed

                        # Format annotation in YOLO format
                        newline = f"{class_id} " + " ".join(map(str, converted_coords))
                        annotations.append(newline)

                # Write the modified annotations back to the file
                with open(file_path, "w") as outfile:
                    outfile.write("\n".join(annotations) + "\n")

print("✅ Annotation conversion completed successfully!")
