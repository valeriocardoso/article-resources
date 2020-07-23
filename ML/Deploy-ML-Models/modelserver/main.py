"""
Model servel scripts

"""

import base64
import json
import os
import sys
import time
import numpy as np
from PIL import Image
import redis

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils

from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request

app = FastAPI()

db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.get("/")
def index():
    return "Hello World!"
