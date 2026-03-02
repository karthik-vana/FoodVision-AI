"""
prepare_dataset.py
------------------
Extracts images from Food_image_Dataset.zip and splits them into:
  - training_data  : 150 images per class (34 classes)
  - valid_data     : 50 images per class  (34 classes)
  - testing_data   : 5 images per class   (34 classes)

Output folder structure:
  image_Dataset/
      training_data/
          Baked Potato/
          Crispy Chicken/
          ...
      valid_data/
          Baked Potato/
          ...
      testing_data/
          Baked Potato/
          ...
"""

import os
import zipfile
import random
import shutil

# ──────────────────────────────────────────────
# SETTINGS  (change these if needed)
# ──────────────────────────────────────────────
ZIP_FILE     = "Food_image_Dataset.zip"          # source zip file
OUTPUT_DIR   = "image_Dataset"                   # new output folder name
TRAIN_COUNT  = 150                               # images per class for training
VALID_COUNT  = 50                                # images per class for validation
TEST_COUNT   = 5                                 # images per class for testing
SEED         = 42                                # for reproducibility

# ──────────────────────────────────────────────
# STEP 1 : Open the zip file
# ──────────────────────────────────────────────
print("Opening zip file ...")
zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ZIP_FILE)
zf = zipfile.ZipFile(zip_path, "r")

# ──────────────────────────────────────────────
# STEP 2 : Find all classes and their images
# ──────────────────────────────────────────────
# Zip structure: "Food Classification dataset/<ClassName>/<image>.jpeg"
# We group images by their class folder name.

class_images = {}          # { "Baked Potato": [entry_name, ...], ... }

for entry in zf.namelist():
    parts = entry.split("/")
    # We need at least 3 parts: root_folder / class / filename
    if len(parts) >= 3 and parts[2]:           # parts[2] = filename
        class_name = parts[1] 
        class_images.setdefault(class_name, []).append(entry)

print(f"Found {len(class_images)} classes:")
for cls in sorted(class_images):
    print(f"  {cls} : {len(class_images[cls])} images")

# ──────────────────────────────────────────────
# STEP 3 : Create the output folder structure
# ──────────────────────────────────────────────
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR)

# Remove old output folder if it exists (fresh start)
if os.path.exists(base_dir):
    print(f"\nRemoving old '{OUTPUT_DIR}' folder ...")
    shutil.rmtree(base_dir)

splits = {
    "training_data": TRAIN_COUNT,
    "valid_data":    VALID_COUNT,
    "testing_data":  TEST_COUNT,
}

# Create folders
for split_name in splits:
    for cls in class_images:
        folder = os.path.join(base_dir, split_name, cls)
        os.makedirs(folder, exist_ok=True)

print(f"\nCreated folder structure inside '{OUTPUT_DIR}/'")

# ──────────────────────────────────────────────
# STEP 4 : Randomly pick & copy images
# ──────────────────────────────────────────────
random.seed(SEED)
total_needed = TRAIN_COUNT + VALID_COUNT + TEST_COUNT   # 205

print("\nSplitting images ...\n")

for cls in sorted(class_images):
    images = class_images[cls]
    random.shuffle(images)                      # randomize order

    available = len(images)
    if available < total_needed:
        print(f"  WARNING: '{cls}' has only {available} images "
              f"(need {total_needed}). Using all available images.")

    # Slice the shuffled list into train / valid / test
    train_imgs = images[0 : TRAIN_COUNT]
    valid_imgs = images[TRAIN_COUNT : TRAIN_COUNT + VALID_COUNT]
    test_imgs  = images[TRAIN_COUNT + VALID_COUNT : TRAIN_COUNT + VALID_COUNT + TEST_COUNT]

    # Helper to extract images into the right folder
    def save_images(img_list, split_name):
        dest_folder = os.path.join(base_dir, split_name, cls)
        for entry_name in img_list:
            filename = os.path.basename(entry_name)
            dest_path = os.path.join(dest_folder, filename)
            # Read from zip and write to disk
            with zf.open(entry_name) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

    save_images(train_imgs, "training_data")
    save_images(valid_imgs, "valid_data")
    save_images(test_imgs,  "testing_data")

    print(f"  {cls:20s} -> train: {len(train_imgs)}, "
          f"valid: {len(valid_imgs)}, test: {len(test_imgs)}")

zf.close()

# ──────────────────────────────────────────────
# STEP 5 : Print summary
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("DONE!  Dataset split summary:")
print("=" * 50)
for split_name, count in splits.items():
    split_path = os.path.join(base_dir, split_name)
    actual_classes = os.listdir(split_path)
    total_files = sum(
        len(os.listdir(os.path.join(split_path, c)))
        for c in actual_classes
    )
    print(f"  {split_name:15s} : {len(actual_classes)} classes, "
          f"{total_files} total images")
print("=" * 50)
print(f"\nOutput location: {base_dir}")
