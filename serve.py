import pickle
from collections import OrderedDict

from flask import Flask, request, jsonify
import numpy as np

from train import COLUMNS, VARIABLES

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return 'OK'

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json(silent=True)
    
    input_x = OrderedDict(zip(COLUMNS, len(COLUMNS)*[0]))

    # some continues can be null
    for v in VARIABLES['cont_nullable']:
        input_x[v] = np.nan

    for k, v in content.items():
        if k in VARIABLES['bool_str']:
            input_x[k] = int(v=='t')
        elif k in VARIABLES['dummy']:
            dummy_key = k + '_' + str(v)
            if dummy_key in input_x:
                input_x[dummy_key] = 1
        else:
            if k in input_x:
                input_x[k] = float(v)
    
    predicted_class = int(model.predict(list(input_x.values()))[0])
    proba = model.predict_proba(list(input_x.values()))[0].tolist()

    return jsonify({'probabilities': proba, 'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
