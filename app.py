from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, flash
import os
import yaml
import joblib
import numpy as np
import json
import logging
import argparse
from werkzeug.utils import secure_filename

from src.get_data import get_data
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder = static_dir, template_folder = template_dir)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    logger = logging.getLogger("predict")
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    # logger.info('> %f' % prediction)

    return prediction

logger = logging.getLogger("predict")
handler  = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
save_path = 'data_given/'
app.config['UPLOAD_FOLDER'] = save_path
@app.route('/', methods = ["GET","POST"])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']   
        filename = secure_filename(f.filename) 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        data_path = 'data_given/test.dat'
        with open(data_path, "r") as fr1:
            trainFile = fr1.readlines()
        
        data = pd.DataFrame(columns=range(100000))
        
        for i in range(len(trainFile)):
            record = np.fromstring(trainFile[i], dtype=int, sep=' ')
            record_bool = [0 for j in range(100000)]
            for col in record:
                record_bool[col-1] = 1
            
            data.loc[i] = record_bool
        pca = sklearnPCA(n_components = 100)
        pca.fit(data)
        PCA_projected_train = pca.transform(data)
        pca_data = pd.DataFrame(PCA_projected_train)
        response = predict(pca_data)
        pca_data.insert(loc=0, column='Predicted_labels', value=response)
        # pca_data['predicted'] = response 
        pca_data.to_csv('predicted_data/predicted.csv')
        flash('Download predicted data!')
        return render_template("index.html")
    else:
         return render_template("index.html")

@app.route('/download')
def download():
    path = 'predicted_data/predicted.csv'
    return send_file(path, as_attachment=True)


if __name__=="__main__":
    app.run(app.run(host='0.0.0.0', port=5000, debug=True))
    