from flask import Flask,url_for,request
from bert_base.client import BertClient
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
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8055)