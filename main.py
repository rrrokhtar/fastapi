
from fastapi import FastAPI, UploadFile, Form
from typing import Union
from enum import Enum
import layoutparser as lp
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


label_map_of_the_model = {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}
model_path = 'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config'
model = lp.Detectron2LayoutModel(model_path, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8], label_map=label_map_of_the_model)


@app.get('/api/health')
def health():
    return {"state": "OK"}


@app.post('/api/detectron')
async def detectron(file: UploadFile):
    if file is None:
        return {"error": "No file was uploaded"}
    try:
        contents = await file.read()
        img = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_ANYCOLOR)
        layout = model.detect(img)
        return layout
    except Exception as e:
        print(e)
        return {"error": "Unexpected error"}

if __name__ == "__main__":
    pass

