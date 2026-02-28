import os
import shutil
import random

random.seed(42)

base_dir = "processed/preprocessed"
output_dir = "yolo_dataset"

images = os.listdir(os.path.join(base_dir, "images"))
random.shuffle(images)

split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for subset, img_list in [("train", train_imgs), ("val", val_imgs)]:
    for img in img_list:
        ts = img.split("_img")[0]
        src_img = os.path.join(base_dir, "images", img)
        src_lbl = os.path.join(base_dir, "labels", f"{ts}.txt")

        dst_img = os.path.join(output_dir, "images", subset, img)
        dst_lbl = os.path.join(output_dir, "labels", subset, f"{ts}.txt")

        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

print("Dataset Split Complete")