from flask import Flask, render_template, request,jsonify

from flask_cors import CORS
import numpy as np

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
def index():
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

if __name__ == '__main__':
    app.run(debug=True)
