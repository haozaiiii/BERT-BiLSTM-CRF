from flask import Flask,url_for,request
from bert_base.client import BertClient
from werkzeug.contrib.fixers import ProxyFix

import json
app = Flask(__name__)
bc = BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER')
@app.route('/bert/check', methods=['POST'])
def check():
    print(str(request.json))
    result = bc.encode([str(request.json['text'])])
    print(result)
    print('ab',bc.encode(['hello world.']))
    t = {
        'result': str(result)
    }
    return json.dumps(t)

@app.route('/test', methods=['POST'])
def check():
    t = {
        'result': 'hello'
    }
    return json.dumps(t)
if __name__ == '__main__':
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='0.0.0.0', port=8055)