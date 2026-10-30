from ultralytics import YOLOE

from pathlib import Path

dataset_images_folder = Path("/home/dk/Projects/point/Pointing_Dataset-main/captures/color")

images = dataset_images_folder.glob("*.png")

test_image = next(images)

model = YOLOE("yoloe-26s-seg.pt")

# "double-decker bus" is not a COCO class; YOLOE resolves it from the words alone
model.set_classes(["Tennis Ball"])

results = model.predict(test_image)
results[0].show()