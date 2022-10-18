# Basic libraries
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.preprocessing import image
from sklearn.base import BaseEstimator, TransformerMixin

# Target size of each image
width_im = 50
height_im = 50
image_size = (width_im, height_im)

def im2array(path):
    img = image.load_img(path, target_size=image_size) #load and resize
    x = image.img_to_array(img) #Value rgb between 0 and 255
    return list(x/255.0)

class preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, paths):
        return np.array([im2array(path) for path in list(paths)])