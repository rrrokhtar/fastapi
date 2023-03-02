
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

