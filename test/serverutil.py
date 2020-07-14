#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import json
import urllib
import http.server
import test.predict as predictt

class HTTPHandle(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global data
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers","Origin,X-Requested-With,Content-Type,Accept")
        self.end_headers()
        if "?" in self.path:
            controller = urllib.parse.unquote(self.path.split('?',1)[0])
            query_str = urllib.parse.unquote(self.path.split('?',1)[1])
            params = urllib.parse.parse_qs(query_str)
            if controller == "/AelxNER":
                text = params["text"][0] if "text" in params else None
                factors = params["factors"][0] if "factors" in params else None
                result = predictt.online_predict(text,factors)
                data = {"code": "0", "result": result}
        else:
            data = {"code": "1", "result": "非法请求"}
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
        print("请求处理完成...")
        return


def startServer():
    host = ('localhost', 8984)
    server = http.server.HTTPServer(host, HTTPHandle)
    print("Starting server,listen at: http://%s:%s" % host)
    server.serve_forever()


if __name__ == '__main__':
    startServer()