import json
import sched
import time
from predict import Predict
from train2 import get_fitted_model
from train_prepare import PrepareTrain
from sklearn import ensemble
from sklearn.model_selection import cross_val_score


from db_connect import DBConnector
import requests

import cPickle as pickle
import pandas as pd
from flask import Flask
from flask import (request,
                   redirect,
                   url_for,
                   session,
                   jsonify,
                   render_template)

app = Flask(__name__, template_folder='../templates', static_folder="../static")

@app.route('/')
def index():
    # return "boo"
    return render_template('index.html')


@app.route('/score', methods=['GET','POST'])
def score():

    d = requests.get(url).json()
    X = pd.DataFrame.from_dict(d, orient='index').T
    X['acct_type'] = 'pr'

    support = copy.append(X)

    y = predict.predict(support)

    db.save_to_db(X, y[-1][0])
    return render_template('/show_json.html', table=X.T.to_html())


@app.route('/dashboard')
def dashboard():

    df = db.read_frm_db()

    history = df[['name','fraud']]
    return render_template('/dashboard.html', table=history.to_html())


if __name__ == '__main__':
    #url = ""

    db = DBConnector()
    data = pd.read_json('data/raw/data.json')
    copy = data.copy()
    X_all, y_all = PrepareTrain(data, undersample=False).prepare_data()
    model = get_fitted_model(X_all, y_all)
    predict = Predict(model)
    app.run(host='0.0.0.0', port=8080, debug=True)
