import requests
import json

def get_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
               }
    response = requests.get(url,headers=headers)
    code = response.status_code
    print(code)
    return code

def post_url(url,post_content):
    headers = {'Content-Type': 'application/json',
               }
    data = {'words': [post_content]}
    response = requests.post(url,data=json.dumps(data),headers=headers,timeout=10)

    return int(json.loads(response.content)['score'])