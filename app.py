from flask import Flask, render_template, request,jsonify

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # mp = tekanan maksimum
    # mu = tekanan rata-rata
    
    mp1 = request.json['mp1']
    mp2 = request.json['mp2']
    mp3 = request.json['mp3']
    mp4 = request.json['mp4']
    mp5 = request.json['mp5']
    mp6 = request.json['mp6']
    
    mu1 = request.json['mu1']
    mu2 = request.json['mu2']
    mu3 = request.json['mu3']
    mu4 = request.json['mu4']
    mu5 = request.json['mu5']
    mu6 = request.json['mu6']
    
    mp = [mp1, mp2, mp3, mp4, mp5, mp6]
    mu = [mu1, mu2, mu3, mu4, mu5, mu6]
        
    return jsonify({
        'message':'success',
        'code':200,
        'data':{
            'mp':mp,
            'mu':mu
        }
    })
if __name__ == '__main__':
    app.run(debug=True)
