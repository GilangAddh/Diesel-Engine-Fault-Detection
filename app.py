from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    section = [
        {
            "name": "Tekanan Maksimal",
            "label": ["Tekanan Maksimal 1", "Tekanan Maksimal 2", "Tekanan Maksimal 3", "Tekanan Maksimal 4", "Tekanan Maksimal 5", "Tekanan Maksimal 6"],
            "input": ["mp1", "mp2", "mp3", "mp4", "mp5", "mp6"],
            "satuan": "Pa",
            "rumus":"mp.png"
        },
        {
            "name": "Tekanan Rata-Rata",
            "label": ["Tekanan Rata-Rata 1", "Tekanan Rata-Rata 2", "Tekanan Rata-Rata 3", "Tekanan Rata-Rata 4", "Tekanan Rata-Rata 5", "Tekanan Rata-Rata 6"],
            "input": ["mu1", "mu2", "mu3", "mu4", "mu5", "mu6"],
            "satuan": "Pa",
            "rumus":"mu.png"
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
            ],
            "satuan": "Hz",
            "rumus":"fr.png"

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
            ],
            "satuan": "Nm",
            "rumus":"amp.png"
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
            ],
            "satuan": "°",
            "rumus":"gh.png"
        }
    ]
    
    isDebug = 'true'
    return render_template('index.html', section=section, isDebug=isDebug)

@app.route("/predict", methods=['POST'])
def predict():
    label_dict = {
        0: 'normal (without faults)',
        1: 'pressure reduction in the intake manifold',
        2: 'compression ratio reduction',
        3: 'reduction of amount of fuel injected'
    }

    data = request.json
    data = data.get('data_item', [])

    if isinstance(data, list):
        data = np.array([float(item) for item in data], dtype=np.float32)
        if np.any(data > 1):
            scaler = MinMaxScaler()
            split_data = np.split(data, [6, 12, 36, 60])        
            normalized_data = [scaler.fit_transform(segment.reshape(-1, 1)).flatten() for segment in split_data]

            data = np.concatenate(normalized_data)
    else:
        return jsonify({'message': 'Invalid input', 'code': 400})
    
    data = np.expand_dims(data, axis=0)
        
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1  
    session_options.inter_op_num_threads = 1 
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 

    model_path = 'model/model.onnx'
    session = ort.InferenceSession(model_path, sess_options=session_options)
        
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: data})

    predicted_class = np.argmax(result[0], axis=1)
    
    predicted_labels = [label_dict[pred] for pred in predicted_class]
    return jsonify({
        'message':'success',
        'code':200,
        "data":predicted_labels[0],
    })

if __name__ == '__main__':
    app.run(debug=True)
