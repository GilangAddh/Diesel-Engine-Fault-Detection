from operator import methodcaller
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    section = [
        {
            "name": "Tekanan Maksimal",
            "label": ["Tekanan Maksimal 1", "Tekanan Maksimal 2", "Tekanan Maksimal 3", "Tekanan Maksimal 4", "Tekanan Maksimal 5", "Tekanan Maksimal 6"],
            "input": ["mp1", "mp2", "mp3", "mp4", "mp5", "mp6"]
        },
        {
            "name": "Tekanan Rata-Rata",
            "label": ["Tekanan Rata-Rata 1", "Tekanan Rata-Rata 2", "Tekanan Rata-Rata 3", "Tekanan Rata-Rata 4", "Tekanan Rata-Rata 5", "Tekanan Rata-Rata 6"],
            "input": ["mu1", "mu2", "mu3", "mu4", "mu5", "mu6"]
        },
        {
            "name": "Frekuensi",
            "label": [
                "Frekuensi 1", "Frekuensi 2", "Frekuensi 3", "Frekuensi 4", "Frekuensi 5", 
                "Frekuensi 6", "Frekuensi 7", "Frekuensi 8", "Frekuensi 9", "Frekuensi 10", 
                "Frekuensi 11", "Frekuensi 12", "Frekuensi 13", "Frekuensi 14", "Frekuensi 15", 
                "Frekuensi 16", "Frekuensi 17", "Frekuensi 18", "Frekuensi 19", "Frekuensi 20", 
                "Frekuensi 21", "Frekuensi 22", "Frekuensi 23", "Frekuensi 24"
            ],
            "input": [
                "fr1", "fr2", "fr3", "fr4", "fr5", "fr6",
                "fr7", "fr8", "fr9", "fr10", "fr11", "fr12", 
                "fr13", "fr14", "fr15", "fr16", "fr17", "fr18",
                "fr19", "fr20", "fr21", "fr22", "fr23", "fr24"
            ]
        },
        {
            "name": "Amplitude",
            "label": [
                "Amplitude 1", "Amplitude 2", "Amplitude 3", "Amplitude 4", "Amplitude 5", 
                "Amplitude 6", "Amplitude 7", "Amplitude 8", "Amplitude 9", "Amplitude 10", 
                "Amplitude 11", "Amplitude 12", "Amplitude 13", "Amplitude 14", "Amplitude 15", 
                "Amplitude 16", "Amplitude 17", "Amplitude 18", "Amplitude 19", "Amplitude 20", 
                "Amplitude 21", "Amplitude 22", "Amplitude 23", "Amplitude 24"
            ],
            "input": [
                "amp1", "amp2", "amp3", "amp4", "amp5", "amp6",
                "amp7", "amp8", "amp9", "amp10", "amp11", "amp12", 
                "amp13", "amp14", "amp15", "amp16", "amp17", "amp18",
                "amp19", "amp20", "amp21", "amp22", "amp23", "amp24"
            ]
        },
        {
            "name": "Gerakan Harmoni",
            "label": [
                "Gerakan 1", "Gerakan 2", "Gerakan 3", "Gerakan 4", "Gerakan 5", 
                "Gerakan 6", "Gerakan 7", "Gerakan 8", "Gerakan 9", "Gerakan 10", 
                "Gerakan 11", "Gerakan 12", "Gerakan 13", "Gerakan 14", "Gerakan 15", 
                "Gerakan 16", "Gerakan 17", "Gerakan 18", "Gerakan 19", "Gerakan 20", 
                "Gerakan 21", "Gerakan 22", "Gerakan 23", "Gerakan 24"
            ],
            "input": [
                "gh1", "gh2", "gh3", "gh4", "gh5", "gh6",
                "gh7", "gh8", "gh9", "gh10", "gh11", "gh12", 
                "gh13", "gh14", "gh15", "gh16", "gh17", "gh18",
                "gh19", "gh20", "gh21", "gh22", "gh23", "gh24"
            ]
        }
    ]
    return render_template('index.html', section=section)

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

    for i in range(1, nSilinder + 1):
        tekanan_values = [float(request.json[f"tekanan{j}s{i}"]) for j in range(1, nPercobaan + 1)]
        tmax = max(tekanan_values)
        tmean = sum(tekanan_values) / nPercobaan
        result[f"mp{i}"] = tmax
        result[f"mu{i}"] = tmean

    return jsonify({
        'message': 'success',
        'code': 200,
        "data": result,
    })

# Untuk sementara predict menggunakan data dummy
@app.route("/predict", methods=['POST'])
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

    #get data form
    # x = tf.convert_to_tensor([0.989485505, 0.989842059, 0.989769292, 0.989689835, 0.858474522, 0.98955769, 
    #                           0.973703785, 0.999920595, 0.999897156, 0.998918769, 0.91455455, 0.999916694, 
    #                           0, 0.04389313, 0.08778626, 0.131679389, 0.173664122, 0.217557252, 0.261450382, 
    #                           0.305343511, 0.347328244, 0.391221374, 0.435114504, 0.479007634, 0.520992366, 
    #                           0.564885496, 0.608778626, 0.652671756, 0.694656489, 0.738549618, 0.782442748, 
    #                           0.826335878, 0.868320611, 0.91221374, 0.95610687, 1, 0.033104492, 0.006437203, 
    #                           0.171023514, 0.018217705, 0.211753935, 0.511552565, 0.148961791, 0.056065341, 
    #                           0.050062193, 0.049112419, 0.351079693, 0.9442107, 0.05124279, 0.007069899, 
    #                           0.174956726, 0.023992546, 0.008563485, 0.146755253, 0.039499209, 0.02358608, 
    #                           0.021458363, 0.005654635, 0.017789763, 0.032057186, 0.689793782, 0.01142033, 
    #                           0.24097166, 0.766363046, 0.045421486, 0.356839364, 0.943136961, 0.864437396, 
    #                           0.226721672, 0.06251139, 0.979988337, 0.25853748, 0.621645889, 0.866603205, 
    #                           0.689587252, 0.735319815, 0.944162944, 0.396483359, 0.831693689, 0.486495649, 
    #                           0.917102811, 0.680523642, 0.454790335, 0.058308235])
    
    data = request.json
    print("Received data:", data)
    data = data.get('data_item', [])

    if isinstance(data, list):
            data = [float(item) for item in data]
    else:
        return jsonify({'message': 'Invalid input', 'code': 400})

    x = tf.convert_to_tensor(data, dtype=tf.float32)
    
    data_tf = tf.expand_dims(x, 0)
    predicted_classes = trained_model.predict(data_tf).argmax(axis=1)
    
    # Map predictions to class labels
    predicted_labels = [label_dict[pred] for pred in predicted_classes]
    return jsonify({
        'message':'success',
        'code':200,
        "data":predicted_labels[0],
    })

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form.getlist('data_item[]')  # menerima input sebagai array
#     # Lakukan sesuatu dengan data
#     return f"Data yang diterima: {data}"

if __name__ == '__main__':
    app.run(debug=True)
