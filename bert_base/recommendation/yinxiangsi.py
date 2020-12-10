from pypinyin import pinyin, lazy_pinyin, Style
import jieba
import numpy as np
import plyvel
import os
import configparser
import sys
# CONFIG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/dict/conf.ini'
# config = configparser.ConfigParser()
# config.read(CONFIG_PATH)
#
# if os.name == 'nt':
#     # ngram_model_path = r'D:\work\bert_ner\leveldb_model\win_bigram_leveldb_tmp'
#     ngram_model_path = config['WIN']['ngram_model_path']
# else:
#     # ngram_model_path = r'/home/panyinghao/model/leveldb/py_level_db3'
#     ngram_model_path = config['LINUX']['ngram_model_path']
#
#
# db = plyvel.DB(ngram_model_path)
# global db
from bert_base.recommendation.DictObject import DictObject, remove_num
from bert_base.recommendation.get_post_url import post_url

dict_obj = DictObject()

global fastPredict
global get_ngram_score_url
global get_ppl_score_url
global ngram_top_n
global ppl_word

import json
import requests
import string

punc = string.punctuation
punctuation = '“”！@#￥%……&*（），。、【】{}：《》？/|'

headers = {'Content-Type': 'application/json',
       }

def sigmoid_derivative(x):
    '''sigmoid'''
    s = 1 / (1 + np.exp(-x))
    # ds = s * (1 - s) #梯度
    return s

def yin_zi_find_pre_after_word(sentence ,replace_index):
    '''
    分词，获取前一单词，后一单词，
    分别组成串返回
    '''
    # 前一单词+当前单词
    pre_word_str = ''
    # 当前单词+后一单词
    after_word_str = ''

    seg_list = jieba.lcut(sentence)
    # print('分词结果 ： ','/'.join(seg_list))
    index = 0
    word_index = 0
    for i, word in enumerate(seg_list):
        if index >= replace_index:
            word_index = i
            break
        index += len(word)

    if word_index > 0:
        pre_word_str = seg_list[i - 1] + seg_list[i]
    if word_index < len(seg_list) - 1:
        after_word_str = seg_list[i] + seg_list[i + 1]

    return pre_word_str, after_word_str

# def get_call_ngram(word_str,ngram_db):
def get_call_ngram(word_str):
    '''
    从ngram中获取字符串出现次数
    '''
    if len(word_str)<1:
        return 0
    # feq = db.get(word_str.encode('utf-8'),0)
    feq = post_url(get_ngram_score_url,word_str)
    # if feq != 0:
    #     feq = int(feq.decode('utf-8'))

    return feq

def get_call_ngram_http(word_str,ngram_db):
    db = ngram_db
    '''
    从ngram中获取字符串出现次数
    '''
    if len(word_str)<1:
        return 0
    feq = db.get(word_str[0].encode('utf-8'),0)
    # if type(feq) == bytes:
    #     feq = feq.decode('utf-8')
    if feq != 0:
        feq = int(feq.decode('utf-8'))
    # feq = post_url(get_ngram_score_url,word_str)
    return feq

def yin_zi_call_ngram(sentence ,replace_index):
    '''
    sentence:替换字后句子
    replace_index:替换字索引
    音相似字替换打分
    '''

    call = 0
    # 前一单词+当前单词, 当前单词+后一单词
    pre_word_str, after_word_str = yin_zi_find_pre_after_word(sentence ,replace_index)
    #向前字符串词频 + sigmoid
    pre_word_str_feq = get_call_ngram(pre_word_str)
    # pre_word_str_feq = sigmoid_derivative(pre_word_str_feq/1000.0)
    #向后字符串词频 + sigmoid
    after_word_str_feq = get_call_ngram(after_word_str)
    # after_word_str_feq = sigmoid_derivative(after_word_str_feq/1000.0)



    call = pre_word_str_feq+after_word_str_feq
    # print('前缀串','ngram词频','后缀串','ngram词频',' : ',pre_word_str+'|',str(pre_word_str_feq)+'|',after_word_str+'|',str(after_word_str_feq)+'|')

    return call

def yin_zi_call_ppl(sentence ,replace_index):

    # 计算ppl的索引位置
    index_set = set()
    for index in replace_index:
        i=1
        while (index-i)>0 and i<ppl_word:
            if sentence[index-i] in punc or sentence[index-i] in punctuation:
                break
            else:
                index_set.add(index-i)
                i+=1
        index_set.add(index)
        j=1
        while (index+j)<len(sentence) and j<ppl_word:
            if sentence[index+j] in punc or sentence[index+j] in punctuation:
                break
            else:
                index_set.add(index+j)
                j+=1
    index_list = list(index_set)
    sen_dict = {'text':sentence,'index_list':index_list}
    response = requests.post(get_ppl_score_url, data=json.dumps(sen_dict), headers=headers)
    score = float(json.loads(response.content)['data']['score'])
    if score is not None:
        return score
    else:
        return -1

    # try:
    #     score = fastPredict.predict_index(sentence, replace_index)
    #
    # #     # return 1/score
    #     return score
    # except Exception as e:
    #     print(e)
    # #     print('ppl except, not score: ',str(score),' || sentence :',sentence)
    #     sys.stdout.flush()
    #     return -1

def yin_zi_call_ppl_estimator(sentence ,replace_index):

    # 计算ppl的索引位置
    index_set = set()
    for index in replace_index:
        i=1
        while (index-i)>0 and i<ppl_word:
            if sentence[index-i] in punc or sentence[index-i] in punctuation:
                break
            else:
                index_set.add(index-i)
                i+=1
        index_set.add(index)
        j=1
        while (index+j)<len(sentence) and j<ppl_word:
            if sentence[index+j] in punc or sentence[index+j] in punctuation:
                break
            else:
                index_set.add(index+j)
                j+=1
    index_list = list(index_set)
    # sen_dict = {'text':sentence,'index_list':index_list}
    # response = requests.post(get_ppl_score_url, data=json.dumps(sen_dict), headers=headers)
    # score = float(json.loads(response.content)['data']['score'])
    # if score is not None:
    #     return score
    # else:
    #     return -1

    try:
        score = fastPredict.predict_index(sentence, index_list)

    #     # return 1/score
        return score
    except Exception as e:
        print(e)
    #     print('ppl except, not score: ',str(score),' || sentence :',sentence)
        sys.stdout.flush()
        return -1



def yin_zi_call(sentence ,replace_index,flag):
    if flag == 'ngram':
        call = yin_zi_call_ngram(sentence ,replace_index)
    elif flag == 'ppl':
        # call = yin_zi_call_ppl(sentence ,replace_index)
        call = yin_zi_call_ppl_estimator(sentence ,replace_index)
    return call

def yin_zi_call_post(predict_dict_list,flag):
    # if flag == 'ngram':
    #     call = yin_zi_call_ngram(sentence ,replace_index)
    if flag == 'ppl':
        url = get_ppl_score_url
        headers = {'Content-Type': 'application/json',
                   }
        data = {'predict_dict_list': predict_dict_list}
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=2)
        call = json.loads(response.content)['predict_score']

    return call

def get_ppl_count_start_end():
    pass
def get_sim_word_call(sim_zi_list, sentence, item,token_zi):
    sim_zi_call_dict_ngram = {}
    # ngram 粗排，取top10
    for zi in sim_zi_list:
        replace_sen = sentence[0:item['start']] + zi + sentence[item['end']:]
        sim_zi_call_dict_ngram[zi] = yin_zi_call(replace_sen, item['start'],'ngram')

    sim_zi_call_sort_ngram = sorted(sim_zi_call_dict_ngram.items(), key=lambda x: x[1], reverse=True)

    lens = len(sim_zi_call_sort_ngram) if len(sim_zi_call_sort_ngram)<int(ngram_top_n) else int(ngram_top_n)

    print('ngram打分：', sim_zi_call_sort_ngram)
    sys.stdout.flush()

        # ppl细排，取top1
    sim_zi_call_dict_ppl = {}
    for i in range(lens):
        k,v = sim_zi_call_sort_ngram[i]
        replace_sen = sentence[0:item['start']] + k + sentence[item['end']:]
        # print('替换句：')
        # print(replace_sen)
        # sim_zi_call_dict_ppl[k] = yin_zi_call(replace_sen, item['start'], 'ppl')
        replace_index = [j for j in range(item['start'],item['end'])]
        sim_zi_call_dict_ppl[k] = yin_zi_call(replace_sen, replace_index, 'ppl')

    # 原文计算ppl：
    if token_zi not in sim_zi_call_dict_ppl:
        replace_sen = sentence[0:item['start']] + token_zi + sentence[item['end']:]
        # sim_zi_call_dict_ppl[token_zi] = yin_zi_call(replace_sen, item['start'], 'ppl')
        replace_index = [j for j in range(item['start'], item['end'])]
        # replace_index = get_ppl_count_start_end()
        sim_zi_call_dict_ppl[token_zi] = yin_zi_call(replace_sen, replace_index, 'ppl')
        # print(token_zi,sim_zi_call_dict_ppl.get(token_zi,-1))

    # sim_zi_call_sort_ppl = sorted(sim_zi_call_dict_ppl.items(), key=lambda x: x[1], reverse=True)
    sim_zi_call_sort_ppl = sorted(sim_zi_call_dict_ppl.items(), key=lambda x: x[1])
    print('ppl打分',sim_zi_call_sort_ppl)
    sys.stdout.flush()



    return sim_zi_call_sort_ppl,sim_zi_call_dict_ppl


def get_sim_word_call_post_url(sim_zi_list, sentence, item, token_zi):
    sim_zi_call_dict_ngram = {}
    # ngram 粗排，取top10
    for zi in sim_zi_list:
        replace_sen = sentence[0:item['start']] + zi + sentence[item['end']:]
        sim_zi_call_dict_ngram[zi] = yin_zi_call(replace_sen, item['start'], 'ngram')

    sim_zi_call_sort_ngram = sorted(sim_zi_call_dict_ngram.items(), key=lambda x: x[1], reverse=True)

    lens = len(sim_zi_call_sort_ngram) if len(sim_zi_call_sort_ngram) < int(ngram_top_n) else int(ngram_top_n)

    print('ngram打分：', sim_zi_call_sort_ngram)
    sys.stdout.flush()

    post_dict = {}

    # ppl细排，取top1
    predict_dict_list = []
    tmp_rep_word_set = set()
    for i in range(lens):
        k, v = sim_zi_call_sort_ngram[i]
        replace_sen = sentence[0:item['start']] + k + sentence[item['end']:]
        # print('替换句：')
        # print(replace_sen)
        # sim_zi_call_dict_ppl[k] = yin_zi_call(replace_sen, item['start'], 'ppl')
        replace_index = [j for j in range(item['start'], item['end'])]
        pre_predict_dict = {'replace_sen':replace_sen,'replace_index':replace_index,'replcace_word':k}
        predict_dict_list.append(pre_predict_dict)
        tmp_rep_word_set.add(k)


    # 原文计算ppl：
    if token_zi not in tmp_rep_word_set:
        replace_sen = sentence[0:item['start']] + token_zi + sentence[item['end']:]
        replace_index = [j for j in range(item['start'], item['end'])]
        pre_predict_dict = {'replace_sen': replace_sen, 'replace_index': replace_index, 'replcace_word': token_zi}
        predict_dict_list.append(pre_predict_dict)
        tmp_rep_word_set.add(token_zi)

    sim_zi_call_dict_ppl = yin_zi_call_post(predict_dict_list, 'ppl')


    # sim_zi_call_sort_ppl = sorted(sim_zi_call_dict_ppl.items(), key=lambda x: x[1], reverse=True)
    sim_zi_call_sort_ppl = sorted(sim_zi_call_dict_ppl.items(), key=lambda x: x[1])
    print('ppl打分', sim_zi_call_sort_ppl)
    sys.stdout.flush()

    return sim_zi_call_sort_ppl, sim_zi_call_dict_ppl

def filter_houxuanci_one_word(sim_zi_call_sort,sim_zi_call_dict):
    '''
    主要根据ppl，去除ppl最低的返回
    '''
    return [sim_zi_call_sort[0][0]]

def filter_houxuanci_three_word(sim_zi_call_sort,sim_zi_call_dict):
    '''
    按排序找到前三个候选词返回，前三个中出现词频差距较大的则删除三个中较后面的
    ***主要根据ngram计算出的词频
    '''
    ret_list = []
    if len(sim_zi_call_sort) > 0:
        ret_i = 0
        for word in sim_zi_call_sort:
            # 词频小于1的不要
            if ret_i < 3 and sim_zi_call_dict.get(word[0],0) > 0:
                ret_list.append(word[0])
                ret_i += 1

    if len(ret_list) > 1:
    # if len(ret_list) > 1 and sim_zi_call_dict[ret_list[0]] != 0:
        # for word in ret_list:
        #     if sim_zi_call_dict[word] == 0:
        #         ret_list.remove(word)

        for i in range(1, len(ret_list)):
            if sim_zi_call_dict[ret_list[i]] * 1000 < sim_zi_call_dict[ret_list[i - 1]]:
                ret_list = ret_list[0:i]
                break

    return ret_list

def yin_zi_ngram(sentence, item):
    '''
    ***找到候选字
    1.根据字找到该字的所有音
    2.根据音找到该音的所有字
    =======
    ***候选字打分取最高
    3.分三种情况：单字成词，与前一个字组成词，与后一个字组成词。分别计算与上下文的前后词共现频率
    '''
    token_zi = item['token']
    # 1.根据字找到该字的所有音
    # yin_list = dict_obj.zi_yin_dict.get(token_zi) #通过字典查找
    yin_list = pinyin(token_zi,style=Style.TONE3,  heteronym=True)[0] # 启用多音字模式
    yin_list = remove_num(yin_list)


    # 2.1根据音找到该音的所有字
    sim_zi_list = []
    for yin in yin_list:
        sim_zi_list += dict_obj.yin_zi_dict.get(yin,[])

    # 2.2 根据音找到相似音所有的字
    sim_zi_list = list(set(sim_zi_list) | dict_obj.sim_yin_zi_dict.get(token_zi,{}))

    # 添加原文元素
    if token_zi not in sim_zi_list:
        # sim_zi_list.remove(token_zi)
        sim_zi_list.add(token_zi)

    # print('音相似字候选集',sim_zi_list)

    # 3.打分
    sim_zi_call_sort,sim_zi_call_dict = get_sim_word_call(sim_zi_list, sentence, item,token_zi)

    # return filter_houxuanci_three_word(sim_zi_call_sort,sim_zi_call_dict)
    return filter_houxuanci_one_word(sim_zi_call_sort,sim_zi_call_dict)

def yin_ci_ngram(sentence, item):
    token_zi = item['token']
    # 1.汉字转拼音
    # yin_list_list = pinyin(token_zi,style=Style.TONE3,  heteronym=True)  # 汉字转拼音
    yin_list_list = pinyin(token_zi,style=Style.TONE3)  # 汉字转拼音，不启用多音字
    yin_list = []
    for yin_element in yin_list_list:
        yin_element = remove_num(yin_element)
        yin_list.append('|'.join(yin_element))
    yin_str = '_'.join(yin_list)

    #2.找到拼音下所有的词语
    sim_zi_list = dict_obj.yin_ci_dict.get(yin_str,set())

    # 移除原文元素
    # if token_zi not in sim_zi_list:
        # sim_zi_list.remove(token_zi)
        # sim_zi_list.add(token_zi)

    # print('音相似词候选集', sim_zi_list)

    # 3.打分
    sim_zi_call_sort,sim_zi_call_dict = get_sim_word_call(sim_zi_list, sentence, item,token_zi)

    # return filter_houxuanci(sim_zi_call_sort,sim_zi_call_dict)
    return filter_houxuanci_one_word(sim_zi_call_sort,sim_zi_call_dict)



def yinxiangsi(sentence, item, ngram_score_url, ppl_score_url, top_n,ppl_word_n):
# def yinxiangsi(sentence, item, ppl_model, ngram_score_url, top_n,ppl_word_n):
    global fastPredict
    global get_ngram_score_url
    global ngram_top_n
    global get_ppl_score_url
    global ppl_word

    # fastPredict = ppl_model
    get_ngram_score_url = ngram_score_url
    ngram_top_n = int(top_n)
    get_ppl_score_url = ppl_score_url
    ppl_word = int(ppl_word_n)


    '''
    分发：字错误  词错误
    返回：建议词汇
    '''
    if len(item['token'])==1:
        return yin_zi_ngram(sentence, item)
    elif len(item['token'])>1:
        return yin_ci_ngram(sentence, item)
    else:
        print('error : error token length is 0')
        return None


def test():
    print(pinyin('中心', heteronym=True))
    print(pinyin('中心'))
    print(pinyin('重心'))
    print(jieba.lcut('粤海化工辩称，对于原告当庭增加诉讼请求不要求新的答辩期限和举证期限。'))
    print(jieba.lcut('粤海化工辨城，对于原告当庭增加诉讼请求不要球新的答辩期限和举证期限。'))
    # feq = db.get('的表现'.encode('utf-8'),0)
    # if feq != 0:
    #     feq = feq.decode('utf-8')
    # print(feq)


def zi_test():
    sentence = r'粤海化工变称，对于原告当庭增加诉讼请求不要求新的答辩期限和举证期限。'
    item = {}
    item['start'] = 4
    item['end'] = 5
    item['errType'] = 0
    item['errType_ch'] = r'音相似'
    item['token'] = r'变'
    item['suggest_word'] = yinxiangsi(sentence, item)


    print('建议词汇', yinxiangsi(sentence, item))

def ci_test():
    # fēng_lì {'锋利', '风栗', '风力', '丰利', '丰丽', '封立'}
    sentence = r'似乎有风栗的冰碴刺中了哈利的心。'
    item = {}
    item['start'] = 3
    item['end'] = 5
    item['errType'] = 0
    item['errType_ch'] = r'音相似'
    item['token'] = r'风栗'

    print('建议词汇', yinxiangsi(sentence, item))
if __name__ == '__main__':
    print('abspath')

    test()
    # ci_test()


