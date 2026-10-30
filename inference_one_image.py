from rfdetr import RFDETRMedium
import supervision as sv
from PIL import Image
import os

dataset_location = os.path.join(os.path.dirname(__file__), "dataset")

test_image = "/home/dk/Projects/point/test2.jpg"
image = Image.open(test_image)

model = RFDETRMedium(pretrain_weights="/home/dk/Projects/point/output/checkpoint_best_ema.pth")

detections = model.predict(image, threshold=0.5)

text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
    "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

bbox_annotator = sv.BoxAnnotator(color=color,thickness=thickness)
label_annotator = sv.LabelAnnotator(
    color=color,
    text_color=sv.Color.BLACK,
    text_scale=text_scale)



detections_image = image.copy()
detections_image = bbox_annotator.annotate(detections_image, detections)

sv.plot_images_grid(images=[detections_image], grid_size=(1, 1), titles=["Detection"])