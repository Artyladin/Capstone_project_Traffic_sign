from flask import Flask, request, redirect, jsonify, render_template, url_for
import pandas as pd
import urllib.request
import os

import joblib
from utils import preprocessor
from keras.preprocessing import image

from keras.models import load_model

app = Flask(__name__)

classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}


upload_folder = ".\\static\\uploads"
app.config['UPLOAD_FOLDER'] = upload_folder

#Load the pipeline (without the model)
with open(r'./Model/pipeline.joblib','rb') as file :
    pipeline = joblib.load(file)

#Load the model
model=load_model('./Model/Simple_model.h5')

#Append the model to the pipeline
pipeline.steps.append(['sequential',model])

@app.route('/')
    
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():

    if request.method == "POST":
        
        img = request.files["image"]
        path=os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        print(path)
        img.save(path)
        img_path=f"{upload_folder}\\{img.filename}"
        predicted_trafic_sign = classes[pipeline.predict(pd.Series(img_path)).argmax(axis=-1)[0]]
        print(predicted_trafic_sign)
    return render_template('index.html', prediction = predicted_trafic_sign, filename = img_path)


if __name__ == "__main__":
    app.run(debug=True)
