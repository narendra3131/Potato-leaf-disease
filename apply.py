from flask import Flask,render_template,request
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing import image  #type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions #type: ignore
import numpy as np
from PIL import Image

categories=["potato early blight","potato healthy","potato late blight"]

# model1 = load_model(r"D:\potato\potato_des\potato_des\Model2.h5")

# model1=pickle.load(open('model2.pkl','rb'))

# with open('model2.pickle', 'rb') as model_file:
#     model1 = pickle.load(model_file)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/formate',methods=['POST'])
def formdata():
    file = request.form['file']
    img = Image.open(file)
    img = img.resize((100,100))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds=model1.predict(x)
    pred = np.argmax(preds, axis=-1)
    # # intent=categories[pred[0]]
    # return  categories[pred[0]]
    # return render_template('index.html', intent=intent)
    
    return render_template('index.html',intent = categories[pred[0]])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)