import os
import json
from PIL import Image
from torchvision.datasets import CIFAR100

OUTPUT_DIR = "cifar3"

SELECTED_CLASSES = [
    "apple",
    "bus",
    "dolphin"
]

for split in ["train", "test"]:
    for class_name in SELECTED_CLASSES:
        path = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(path, exist_ok=True)

train_dataset = CIFAR100(
    root="data",
    train=True,
    download=True
)

test_dataset = CIFAR100(
    root="data",
    train=False,
    download=True
)

class_to_idx = train_dataset.class_to_idx

print("Сохраняем train...")

train_counters = {cls: 0 for cls in SELECTED_CLASSES}

for image, label in train_dataset:

    class_name = train_dataset.classes[label]

    if class_name not in SELECTED_CLASSES:
        continue

    index = train_counters[class_name]

    save_path = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name,
        f"{index}.png"
    )

    image.save(save_path)

    train_counters[class_name] += 1

print("Сохраняем test...")

test_counters = {cls: 0 for cls in SELECTED_CLASSES}

for image, label in test_dataset:

    class_name = test_dataset.classes[label]

    if class_name not in SELECTED_CLASSES:
        continue

    index = test_counters[class_name]

    save_path = os.path.join(
        OUTPUT_DIR,
        "test",
        class_name,
        f"{index}.png"
    )

    image.save(save_path)

    test_counters[class_name] += 1

format_data = {
    "dataset": "CIFAR100",
    "selected_classes": SELECTED_CLASSES,
    "class_to_index": {
        cls_name: idx
        for idx, cls_name in enumerate(SELECTED_CLASSES)
    },
    "image_size": [32, 32],
    "channels": 3,
    "train_images_per_class": train_counters,
    "test_images_per_class": test_counters
}

json_path = os.path.join(OUTPUT_DIR, "format.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(format_data, f, indent=4, ensure_ascii=False)

print("\nГотово!")
print(f"Датасет сохранен в папке: {OUTPUT_DIR}")