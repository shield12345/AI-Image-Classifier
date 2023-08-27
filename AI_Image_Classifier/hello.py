from flask import Flask
from flask import render_template, request
import os
from PIL import Image
import numpy as np
from keras.models import load_model
import sklearn
import pickle
import pandas as pd
#from tensorflow.keras.optimizers import Adam, Adamax

from keras import backend as K

# custom object to load the model using keras
def F1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

app = Flask(__name__)

# load the model, and pass in the custom metric function
model = load_model('Real vs Fake Images-2-(32 X 32)- 95.96.h5', custom_objects={"F1_score": F1_score })

@app.route("/")
def about():
    return render_template("index.html")

def predict_image(img_path):
    img = Image.open(img_path)
    img = img.resize((32, 32))
    img = np.array(img)
    img = img / 255.0 
    img = np.expand_dims(img, axis = 0)
    prediction = model.predict(img)
    class_names = ['Real', 'Fake']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    prediction = None
    img_path = None
    if request.method == 'POST':
        uploaded_image = request.files['my_image']
        # check any file is uploaded or not
        print(uploaded_image)
        if uploaded_image.filename != '':
            img_path = "static/" + uploaded_image.filename
            uploaded_image.save(img_path)
        
        prediction = predict_image(img_path)
        
    return render_template("index.html", prediction = prediction, img_path = img_path)

if __name__ =='_main_':
	#app.run(debug = True)
	app.run(debug=True)
    