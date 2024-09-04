from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/getMaxMean", methods=['POST'])
def index():
    nPercobaan = int(request.json['nPercobaan'])
    nSilinder = int(request.json.get('nSilinder', 6))
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

if __name__ == '__main__':
    app.run(debug=True)
