from flask import Flask, render_template, request
from cv2 import cv2
from keras.models import load_model
import numpy as np
model = load_model('model_FTVGG16.h5')
print('Loading model_FTVGG16 model')

class_names = ['Banh_Chung', 'Banh_Mi', 'Banh_Pia', 'Banh_Trang_Nuong',
               'Banh_Xeo', 'Bun_Dau_Mam_Tom', 'Ca_Kho', 'Chao_Long', 'Com_Tam', 'Pho']
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html', data="", class_names=enumerate(class_names), percents=None)


@app.route('/after', methods=['GET', 'POST'])
def after():

    global model, class_names, model_vgg16

    img = request.files['file_image']
    img.save('static/image/file.jpg')
    image = cv2.imread('static/image/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.reshape(image, (1, 150, 150, 3))
    prediction = model.predict(image)
    pred_labels = np.argmax(prediction, axis=1)
    prediction = [round(float(prediction[0][i]), 3)
                  for i in range(len(class_names))]
    final = class_names[int(pred_labels)]
    return render_template('index.html', data=final.replace('_', ' '), class_names=enumerate(class_names), percents=prediction)


if __name__ == "__main__":
    app.run(debug=True)
