import os

from rfdetr import RFDETRMedium

model = RFDETRMedium()

dataset_location = os.path.join(os.path.dirname(__file__), "dataset")

model.train(dataset_dir=dataset_location, epochs=25, batch_size=4, grad_accum_steps=4)

