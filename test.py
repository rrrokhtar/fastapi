import layoutparser as lp
import numpy as np
import cv2
import os

image = cv2.imread("./image_0.jpg")
# Convert the image from BGR (cv2 default loading style) to RGB
image = image[..., ::-1]
label_map_of_the_model = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}
model_path = 'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config'
model = lp.Detectron2LayoutModel(model_path,
                                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                label_map=label_map_of_the_model)

# Detect the layout of the input image
layout = model.detect(image)
print(layout)
lp.draw_box(image, layout, box_width=3)
