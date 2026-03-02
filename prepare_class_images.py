import os
from PIL import Image

source_dir = r"c:\Users\LENOVO\Downloads\Task-2\image_Dataset\training_data"
dest_dir = r"c:\Users\LENOVO\Downloads\Task-2\static\images\classes"

os.makedirs(dest_dir, exist_ok=True)

for root, dirs, files in os.walk(source_dir):
    if root == source_dir:
        for class_name in dirs:
            class_path = os.path.join(root, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if images:
                src_image_path = os.path.join(class_path, images[0])
                dest_image_path = os.path.join(dest_dir, f"{class_name}.jpg")
                try:
                    img = Image.open(src_image_path).convert('RGB')
                    min_dim = min(img.size)
                    left = (img.width - min_dim) / 2
                    top = (img.height - min_dim) / 2
                    right = (img.width + min_dim) / 2
                    bottom = (img.height + min_dim) / 2
                    img = img.crop((left, top, right, bottom))
                    img = img.resize((150, 150), Image.Resampling.LANCZOS)
                    img.save(dest_image_path, "JPEG", quality=85)
                    print(f"Successfully processed {class_name}")
                except Exception as e:
                    print(f"Error processing {class_name}: {e}")
