from operator import methodcaller
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

'''
Param :
    nPercobaan : jumlah percobaan (Number/String)
    nSilinder : banyaknya silinder mesin (optional, default : 6)
    tekananN : tekenan ke-n (Number/String)
Contoh :
    Jika percobaan 3 kali, dilakukan di silinder 1 maka :
    nPercobaan = 3,
    tekenan1s1 : 0.75,
    tekenan2s1 : 0.25,
    tekenan3s1 : 0.5,
    
    Jika percobaan 2 kali, dilakukan di silinder 6 maka :
    nPercobaan = 2,
    tekenan1s6 : 5,
    tekenan2s6 : 5,
    
Result : {
    "mp1": Number/String,
    "mp2": Number/String,
    "mu1": Number/String,
    "mu2": Number/String,
    }
'''
@app.route("/getMaxMean",  methods=['POST'])
def getMaxMean():
    nPercobaan = request.json['nPercobaan']
    nSilinder = request.json.get('nSilinder',6)
    result = {}

    for i in range(1,nSilinder+1):
        tekanan_values = [request.json[f"tekanan{j}s{i}"] for j in range(1, nPercobaan+ 1)]
        tmax = max(tekanan_values)
        tmean = sum(tekanan_values) / request.json["nPercobaan"]
        result[f"mp{i}"] = tmax
        result[f"mu{i}"] = tmean

    return jsonify({
        'message':'success',
        'code':200,
        "data":result,
    })

# Untuk sementara predict menggunakan data dummy
@app.route("/predict", methods=['GET'])
def predict():
    model_save_path = f'model/0db_KFold/model_ann_fold_4.h5'
    trained_model = tf.keras.models.load_model(model_save_path)

    # Make predictions
    label_dict = {
        0: 'normal (without faults)',
        1: 'pressure reduction in the intake manifold',
        2: 'compression ratio reduction',
        3: 'reduction of amount of fuel injected'
    }
    
    x = tf.convert_to_tensor([0.989485505, 0.989842059, 0.989769292, 0.989689835, 0.858474522, 0.98955769, 
                              0.973703785, 0.999920595, 0.999897156, 0.998918769, 0.91455455, 0.999916694, 
                              0, 0.04389313, 0.08778626, 0.131679389, 0.173664122, 0.217557252, 0.261450382, 
                              0.305343511, 0.347328244, 0.391221374, 0.435114504, 0.479007634, 0.520992366, 
                              0.564885496, 0.608778626, 0.652671756, 0.694656489, 0.738549618, 0.782442748, 
                              0.826335878, 0.868320611, 0.91221374, 0.95610687, 1, 0.033104492, 0.006437203, 
                              0.171023514, 0.018217705, 0.211753935, 0.511552565, 0.148961791, 0.056065341, 
                              0.050062193, 0.049112419, 0.351079693, 0.9442107, 0.05124279, 0.007069899, 
                              0.174956726, 0.023992546, 0.008563485, 0.146755253, 0.039499209, 0.02358608, 
                              0.021458363, 0.005654635, 0.017789763, 0.032057186, 0.689793782, 0.01142033, 
                              0.24097166, 0.766363046, 0.045421486, 0.356839364, 0.943136961, 0.864437396, 
                              0.226721672, 0.06251139, 0.979988337, 0.25853748, 0.621645889, 0.866603205, 
                              0.689587252, 0.735319815, 0.944162944, 0.396483359, 0.831693689, 0.486495649, 
                              0.917102811, 0.680523642, 0.454790335, 0.058308235])
    x = tf.expand_dims(x, 0)
    predicted_classes = trained_model.predict(x).argmax(axis=1)
    
    # Map predictions to class labels
    predicted_labels = [label_dict[pred] for pred in predicted_classes]
    return jsonify({
        'message':'success',
        'code':200,
        "data":predicted_labels[0],
    })

if __name__ == '__main__':
    app.run(debug=True)
