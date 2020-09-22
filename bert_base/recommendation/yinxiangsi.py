from bert_base.recommendation.DictObject import DictObject, remove_num
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
global db

dict_obj = DictObject()



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

def get_call_ngram(word_str):
    global db
    '''
    从ngram中获取字符串出现次数
    '''
    if len(word_str)<1:
        return 0
    feq = db.get(word_str.encode('utf-8'),0)
    if feq != 0:
        feq = int(feq.decode('utf-8'))

    return feq

def yin_zi_call(sentence ,replace_index):
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
    pre_word_str_feq = sigmoid_derivative(pre_word_str_feq/1000.0)
    #向后字符串词频 + sigmoid
    after_word_str_feq = get_call_ngram(after_word_str)
    after_word_str_feq = sigmoid_derivative(after_word_str_feq/1000.0)



    call = pre_word_str_feq+after_word_str_feq
    # print('前缀串','ngram词频','后缀串','ngram词频',' : ',pre_word_str+'|',str(pre_word_str_feq)+'|',after_word_str+'|',str(after_word_str_feq)+'|')

    return call

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


    # 2.根据音找到该音的所有字
    sim_zi_list = []
    for yin in yin_list:
        sim_zi_list += dict_obj.yin_zi_dict.get(yin,[])
    # 移除原文元素
    if token_zi in sim_zi_list:
        sim_zi_list.remove(token_zi)

    # 移除原文元素
    if token_zi in sim_zi_list:
        sim_zi_list.remove(token_zi)

    # print('音相似字候选集',sim_zi_list)

    # 3.打分
    sim_zi_call_dict = {}
    for zi in sim_zi_list:
        replace_sen = sentence[0:item['start']]+zi+sentence[item['end']:]
        sim_zi_call_dict[zi] = yin_zi_call(replace_sen,item['start'])

    sim_zi_call_sort = sorted(sim_zi_call_dict.items(),key=lambda x:x[1],reverse=True)
    # print('替换词打分：',sim_zi_call_sort)
    # return sim_zi_call_sort[0][0]

    ret_list = []
    if len(sim_zi_call_sort)>0:
        ret_i = 0
        for word in sim_zi_call_sort:
            if ret_i<2:
                ret_list.append(word[0])
                ret_i += 1


    return ret_list


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
    sim_zi_list = dict_obj.yin_ci_dict.get(yin_str,[])
    #移除原文元素
    if token_zi in sim_zi_list:
        sim_zi_list.remove(token_zi)

    # print('音相似词候选集', sim_zi_list)

    # 3.打分
    sim_zi_call_dict = {}
    for zi in sim_zi_list:
        replace_sen = sentence[0:item['start']]+zi+sentence[item['end']:]
        sim_zi_call_dict[zi] = yin_zi_call(replace_sen,item['start'])

    sim_zi_call_sort = sorted(sim_zi_call_dict.items(), key=lambda x: x[1], reverse=True)
    # print('替换词打分：',sim_zi_call_sort)
    # return sim_zi_call_sort[0][0]
    ret_list = []
    if len(sim_zi_call_sort)>0:
        ret_i = 0
        for word in sim_zi_call_sort:
            if ret_i<2:
                ret_list.append(word[0])
                ret_i += 1


    return ret_list



def yinxiangsi(sentence, item, leveldb):
    global db
    db = leveldb

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
    feq = db.get('的表现'.encode('utf-8'),0)
    if feq != 0:
        feq = feq.decode('utf-8')
    print(feq)


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


