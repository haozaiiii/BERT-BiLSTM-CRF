import requests
import json
import ast
import time

url = 'http://192.168.1.121:8055/ner_predicts_service'

correct1_error2_rate = {}
error1_correct2_rate = {}
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
        for one_err_dict in return_dict['result'][0]['items']:

            # if one_err_dict['token'] == one_err_dict['suggest_word'] :
            if one_err_dict['token'] in one_err_dict['suggest_word'] :
                print('原文==建议 ： ', one_err_dict['token'], one_err_dict['suggest_word'])
            try:
                # if one_err_dict['suggest_word'] == str(correct_sentence)[int(one_err_dict['start']):int(one_err_dict['end'])]:
                if str(correct_sentence)[int(one_err_dict['start']):int(one_err_dict['end'])] == list(one_err_dict['suggest_word'])[0][0] :
                    rate = (list(one_err_dict['suggest_word'])[0][1]+1.0) / int(list(one_err_dict['suggest_word'])[1][1]+1.0)
                    if rate not in correct1_error2_rate:
                        correct1_error2_rate[rate] = 0
                    correct1_error2_rate[rate] = correct1_error2_rate[rate]+1

                    correct_count += 1
                    # print(list(one_err_dict['suggest_word'])[0][1],int(list(one_err_dict['suggest_word'])[1][1]))
                    print('识别正确：', index, one_err_dict['suggest_word'],
                          str(correct_sentence)[int(one_err_dict['start']):int(one_err_dict['end'])], ' : ',
                          error_sentence)
                elif str(correct_sentence)[int(one_err_dict['start']):int(one_err_dict['end'])] == list(one_err_dict['suggest_word'])[1][0]:
                    rate = (list(one_err_dict['suggest_word'])[0][1] + 1.0) / int(
                        list(one_err_dict['suggest_word'])[1][1] + 1.0)
                    if rate not in error1_correct2_rate:
                        error1_correct2_rate[rate] = 0
                    error1_correct2_rate[rate] = error1_correct2_rate[rate] + 1

                    correct_count += 1
                    print('识别正确：', index, one_err_dict['suggest_word'],
                          str(correct_sentence)[int(one_err_dict['start']):int(one_err_dict['end'])], ' : ', error_sentence)
                else:
                    error_count += 1
                    # try:
                    #     print('识别错误：',index, one_err_dict['suggest_word'],
                    #           str(correct_sentence)[int(one_err_dict['start']):int(one_err_dict['end'])],' : ',error_sentence)
                    # except:
                    #     print('识别错误错误：',index, one_err_dict,correct_sentence,error_sentence)
            except:
                continue

        index += 1

    print('=====================================')
    correct1_error2_rate_sorted = sorted(correct1_error2_rate.items(),reverse=True)
    correct1_error2_sum = 0
    for k,v in correct1_error2_rate.items():
        correct1_error2_sum += k

    print('correct1_error2 length :', len(correct1_error2_rate_sorted))
    print('correct1_error2 dict :', correct1_error2_rate_sorted)
    print('correct1_error2 average :', correct1_error2_sum*1.0/len(correct1_error2_rate_sorted))

    error1_correct2_rate_sorted = sorted(error1_correct2_rate.items(), reverse=True)
    error1_correct2_sum = 0
    for k,v in error1_correct2_rate.items():
        error1_correct2_sum += k
    print('error1_correct2 length :', len(error1_correct2_rate_sorted))
    print('error1_correct2 dict :', error1_correct2_rate_sorted)
    print('correct1_error2 average :', error1_correct2_sum * 1.0 / len(error1_correct2_rate_sorted))
    print('=====================================')

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
    path = r'D:\work\bert_ner\中文数据\xianshang\yin100w50pTest\biaozhu_yinxiangsi.txt'
    predict(path)