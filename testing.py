import os
from collections import Counter

dataset_path = "artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

class_counts = {}
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        num_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = num_images

print("📊 Class distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")

total = sum(class_counts.values())
print(f"\nTotal images: {total}")
