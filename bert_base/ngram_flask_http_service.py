#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#@Time    : ${DATE} ${TIME}
# @Author  : MaCan (ma_cancan@163.com)
# @File    : ${NAME}.py
"""
# /home/panyinghao/anaconda3/envs/tf13/bin/gunicorn -w 2 -b 0.0.0.0:8055 -k gevent gunciton_run:app
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import flask
from flask import request, jsonify
from flask import make_response
from werkzeug.contrib.fixers import ProxyFix

import sys
import json

from bert_base.recommendation.recommendcore import get_suggest_word

sys.path.append('../..')
import re
import configparser
import plyvel

app = flask.Flask(__name__)

# CONFIG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/dict/conf.ini'
CONFIG_PATH = os.path.dirname(os.path.abspath(__file__))+r'/bert_base/dict/conf.ini'
config = configparser.ConfigParser()
config.read(CONFIG_PATH)



if os.name == 'nt':
    # ngram_model_path = r'D:\work\bert_ner\leveldb_model\win_bigram_leveldb_tmp'
    ngram_model_path = config['WIN']['ngram_model_path']
else:
    # ngram_model_path = r'/home/panyinghao/model/leveldb/py_level_db3'
    ngram_model_path = config['LINUX']['ngram_model_path']


db = plyvel.DB(ngram_model_path)

@app.route('/ngram_suggests_service', methods=['POST'])
def ner_predicts_service():
    sen_dict = get_suggest_word(request.json,db)
    return json.dumps(sen_dict)



if __name__ == "__main__":
    app.wsgi_app = ProxyFix(app.wsgi_app)
    if os.name == 'nt':
        app.run(host='0.0.0.0', port=8045)
    else:
        app.run(host='0.0.0.0', port=8045, processes=True)
        # app.run(processes=True)
    # online_predict()

