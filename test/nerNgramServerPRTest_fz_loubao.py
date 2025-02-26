import requests
import json
import ast
import time

# url = 'http://192.168.1.121:8055/ner_predicts_service'
url = 'http://119.6.127.9:8053/ner_predicts_service'

def post_url(url, post_content):
    global x,status_code_list
    headers = {'Content-Type': 'application/json',
               }
    data = {'sentences': [post_content]}
    response = requests.post(url,data=json.dumps(data),headers=headers,timeout=100)
    return response

def predict(path):
    pos_count = 0
    correct_count = 0
    error_count = 0
    zi_count = 0
    with open(path,'r',encoding='utf-8') as file:
        content = file.read()
    contentList = content.replace('\r','').split('\n')
    index = 0
    start_time = time.time()
    for line in contentList:
        if index>500:
            break

        if len(line.split('\t')) != 3:
            print('error_line : ', line)
            continue
        err_index, correct_sentence, error_sentence = line.split('\t')
        zi_count += len(error_sentence)
        pos_count += len(str(err_index).split(';'))
        response = post_url(url,error_sentence)
        return_dict = ast.literal_eval(response.content.decode('utf-8'))

        if return_dict['code'] == -1:
            print('error line : ', line)
            continue
        for one_err_dict in return_dict['result'][0]['items']:
            if 'suggest_word' not in one_err_dict:
                error_count += 1
                print('文本过长：',str(correct_sentence[int(one_err_dict['start']): int(one_err_dict['end'])]))
            # elif len(one_err_dict['suggest_word']) >0 and one_err_dict['suggest_word'][0] == correct_sentence[int(one_err_dict['start']):int(one_err_dict['end'])]:
            elif len(one_err_dict['suggest_word']) >0 and (one_err_dict['suggest_word'][0] in correct_sentence or correct_sentence in one_err_dict['suggest_word'][0]):
                correct_count += 1
            # if one_err_dict['token'] in one_err_dict['suggest_word'] :
            #     print('原文==建议 ： ',str(one_err_dict['suggest_word'][0]),
            #           str(correct_sentence[int(one_err_dict['start']): int(one_err_dict['end'])]))
            elif len(one_err_dict['suggest_word']) == 0 and (one_err_dict['suggest_word'][0] in correct_sentence or correct_sentence in one_err_dict['suggest_word'][0]):
                correct_count += 1
                print('无建议词汇且预测正确', str(one_err_dict['token']) , correct_sentence)

            else:
                error_count += 1
                print('return_dict', return_dict)
                print('原文!=建议 ： ', str(one_err_dict['suggest_word']), correct_sentence)


        index += 1

    end_time = time.time()
    R = correct_count*1.0/pos_count
    P = correct_count*1.0/(correct_count+error_count)
    print('识别正确错误数：', correct_count)
    print('识别错误数：',(correct_count+error_count))
    print('文中错误：',pos_count)
    print('召回率：',str(R))
    print('精确率：',str(P))
    print('f1 : ',str(2.0*P*R/(P+R)))
    print('耗时%s秒'%str(end_time-start_time))
    print('每秒处理%s字'%str(zi_count*1.0/(end_time-start_time)))




if __name__ == '__main__':
    # path = r'D:\work\bert_ner\中文数据\xianshang\yin100w50pTest\biaozhu_yinxiangsi.txt'
    path = r'loubao.txt'
    predict(path)