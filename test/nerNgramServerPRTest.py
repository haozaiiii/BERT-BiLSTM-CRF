import requests
import json
import ast
import time

url = 'http://192.168.1.121:8055/ner_predicts_service'
# url = 'http://119.6.127.9:8053/ner_predicts_service'

# url = r'http://119.6.127.8:8055/ner_predicts_service'

def post_url(url, post_content):
    global x,status_code_list
    headers = {'Content-Type': 'application/json',
               }
    data = {'sentences': [post_content]}
    response = requests.post(url,data=json.dumps(data),headers=headers,timeout=100)
    return response

def predict(path, out_path):

    outc = open(out_path,'w',encoding='utf-8')
    out_list  = []
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
        if len(err_index)>0:
            pos_count += len(str(err_index).split(';'))
        response = post_url(url,error_sentence)
        return_dict = ast.literal_eval(response.content.decode('utf-8'))

        if return_dict['code'] == -1:
            print('error line : ', line)
            continue

        if len(return_dict['result'][0]['items']) == 0 and len(err_index) == 0:
            # correct_count += 1
            # print('原文正确',error_sentence)
            continue
        elif len(return_dict['result'][0]['items']) == 0 and len(err_index) != 0:
            error_count += 1
        for one_err_dict in return_dict['result'][0]['items']:
            if 'suggest_word' not in one_err_dict:
                error_count += 1
                # print('文本过长：',str(correct_sentence[int(one_err_dict['start']): int(one_err_dict['end'])]))
            elif len(one_err_dict['suggest_word']) >0 and one_err_dict['suggest_word'][0] == correct_sentence[int(one_err_dict['start']):int(one_err_dict['end'])]:
                correct_count += 1
                # print('原文==建议',str(one_err_dict['suggest_word'][0]),error_sentence)
                # out_list.append('原文==建议'+str(one_err_dict['suggest_word'][0])+error_sentence)

            # if one_err_dict['token'] in one_err_dict['suggest_word'] :
            #     print('原文==建议 ： ',str(one_err_dict['suggest_word'][0]),
            #           str(correct_sentence[int(one_err_dict['start']): int(one_err_dict['end'])]))
            elif  one_err_dict['token'] != correct_sentence[int(one_err_dict['start']):int(one_err_dict['end'])]:
                correct_count += 1
                print('建议错误且预测正确', str(one_err_dict['token']) , correct_sentence[int(one_err_dict['start']):int(one_err_dict['end'])])
                out_list.append(str(error_sentence))
                out_list.append('建议错误且预测正确'+str(one_err_dict['token'])+str(correct_sentence[int(one_err_dict['start']):int(one_err_dict['end'])]))
            else:
                error_count += 1
                print(return_dict['result'][0])
                out_list.append(str(return_dict['result'][0]))
                try:
                    print('原文!=建议 ： ', str(one_err_dict['suggest_word'][0]), str(correct_sentence[int(one_err_dict['start']): int(one_err_dict['end'])]))
                    out_list.append('原文!=建议 ： '+str(one_err_dict['suggest_word'][0])+' '+ str(correct_sentence[int(one_err_dict['start']): int(one_err_dict['end'])]))
                except:
                    print('原文!=建议 ： ', str(correct_sentence))
                    out_list.append('原文!=建议 ： '+ str(correct_sentence))


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

    out_list.append('识别正确错误数：'+str(correct_count))
    out_list.append('识别错误数：'+str((correct_count+error_count)))
    out_list.append('文中错误：'+str(pos_count))
    out_list.append('召回率：'+str(str(R)))
    out_list.append('精确率：'+str(str(P)))
    out_list.append('f1 : '+str(str(2.0*P*R/(P+R))))
    out_list.append('耗时%s秒'%str(end_time-start_time))
    out_list.append('每秒处理%s字'%str(zi_count*1.0/(end_time-start_time)))

    for line in out_list:
        outc.write(line)
        outc.write('\r\n')


    outc.close()




if __name__ == '__main__':
    # path = r'D:\work\bert_ner\中文数据\xianshang\yin100w50pTest\biaozhu_yinxiangsi.txt'
    # path = r'D:\work\测试集\法律测试数据\test_formate_data\biaozhu_yinxiangsi.txt'
    path = r'D:\work\测试集\法律测试数据\法律.txt'
    # path = r'D:\work\测试集\法律测试数据\机构名.txt'

    out_path = r'D:\work\测试集\法律测试数据\test_result\121_山东.txt'
    predict(path, out_path)