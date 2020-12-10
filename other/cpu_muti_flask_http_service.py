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

import requests
import os
import flask
from flask import request, jsonify
import json
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
from flask import make_response
from werkzeug.contrib.fixers import ProxyFix

import sys

from bert_base.recommendation.recommendcore import get_suggest_word

sys.path.append('../..')
from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
import re
import configparser

app = flask.Flask(__name__)

CONFIG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/dict/conf.ini'
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

if os.name == 'nt':
    # model_dir = r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\output'
    # bert_dir = r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\uncased_L-12_H-768_A-12'
    bert_dir = config['WIN']['bert_dir']
    model_dir = config['WIN']['model_dir']
    ngram_suggest_url = config['WIN']['ngram_suggest_url']
else:
    bert_dir = config['LINUX']['bert_dir']
    model_dir = config['LINUX']['model_dir']
    ngram_suggest_url = config['LINUX']['ngram_suggest_url']
    if str(config['LINUX']['use_gpu']).lower() == 'true':
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']['CUDA_VISIBLE_DEVICES']



is_training=False
use_one_hot_embeddings=False
batch_size=1
max_seq_length = 202



'''
gpu_config.inter_op_parallelism_threads 一般设置为2
'''

#cpu_num = int(os.environ.get('CPU_NUM', 1))
os.environ['CUDA_VISIBLE_DEVICES']='-1'
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
#gpu_config.device_count={"CPU": cpu_num}
#gpu_config.intra_op_parallelism_threads = cpu_num
#gpu_config.inter_op_parallelism_threads = 2



sess=tf.Session(config=gpu_config)
model=None
yingshe_dict = {"B-yinxiangsi":0,"B-xingxiangsi":1,"B-duozi":2,"B-diandao":3,"B-shaozi":4}
yingshe_ch_dict = {"B-yinxiangsi": '音相似', "B-xingxiangsi": '形相似', "B-duozi": '多字', "B-diandao": '颠倒', "B-shaozi": '少字'}

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1


graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0,num_layers=1,lstm_size=128)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=True)

@app.route('/ner_predicts_service', methods=['POST'])
def ner_predicts_service():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        result = {}
        try:
            ret_dict = {}
            ret_list = []
            total = 0
            sentences = list(request.json['sentences'])
            for sentence in sentences:
                sen_dict = {}
                pre_sen = sentence

                sentence = re.sub('[0-9]','*',sentence)

                if len(sentence) < 2:
                    err_index_end_list = []
                else:
                    sentence = tokenizer.tokenize_No_wordpiece(sentence)
                    input_ids, input_mask, segment_ids, label_ids = convert(sentence)
                    feed_dict = {input_ids_p: input_ids,
                                 input_mask_p: input_mask}
                    # run session get current feed_dict result
                    pred_ids_result = sess.run([pred_ids], feed_dict)
                    pred_label_result = convert_id_to_label(pred_ids_result, id2label)
                    print('原文 : ',str(pre_sen).lower())
                    print('token : ',sentence)
                    print('tag : ',pred_label_result[0])
                    sys.stdout.flush()
                    err_index_end_list = find_index_int_token_tag(str(pre_sen).lower(), sentence, pred_label_result[0])

                sen_dict['sentence'] = pre_sen
                sen_dict['items'] = err_index_end_list
                sen_dict['suggest_word'] = get_suggest_word_http_request(sen_dict)
                total += len(err_index_end_list)
                ret_list.append(sen_dict)
            ret_dict['total'] = total
            ret_dict['result'] = ret_list
            ret_dict['code'] = 0
            rst = make_response(json.dumps(ret_dict))
            return rst, 200, {'Content-Type': 'application/json'}

           # return json.dumps(ret_dict)
        except:
            result['code'] = -1
            result['data'] = 'special character in content'
            result['result'] = []

        return json.dumps(result)

@app.route('/ner_predicts_service_mutisen', methods=['POST'])
def ner_predicts_service_mutisen():

    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        result = {}
        result['code'] = 0
        try:
            ret_dict = {}
            ret_list = []
            total = 0
            sentences = list(request.json['sentences'])
            sentences = cut_sentences(sentences[0])

            for sentence in sentences:
                sen_dict = {}
                pre_sen = sentence

                sentence = re.sub('[0-9]', '*', sentence)

                if len(sentence) < 2:
                    err_index_end_list = []
                else:
                    sentence = tokenizer.tokenize_No_wordpiece(sentence)
                    input_ids, input_mask, segment_ids, label_ids = convert(sentence)
                    feed_dict = {input_ids_p: input_ids,
                                 input_mask_p: input_mask}
                    # run session get current feed_dict result
                    pred_ids_result = sess.run([pred_ids], feed_dict)
                    pred_label_result = convert_id_to_label(pred_ids_result, id2label)
                    print(str(pre_sen).lower())
                    print(sentence)
                    print(pred_label_result[0])
                    sys.stdout.flush()
                    err_index_end_list = find_index_int_token_tag(str(pre_sen).lower(), sentence, pred_label_result[0])

                sen_dict['sentence'] = pre_sen
                sen_dict['items'] = err_index_end_list
                sen_dict['suggest_word'] = get_suggest_word_http_request(sen_dict)
                total += len(err_index_end_list)
                ret_list.append(sen_dict)
            ret_dict['total'] = total
            ret_dict['result'] = ret_list
            ret_dict['code'] = 0
            rst = make_response(json.dumps(ret_dict))
            return rst, 200, {'Content-Type': 'application/json'}

           # return json.dumps(ret_dict)
        except:
            result['code'] = -1
            result['data'] = 'special character in content'
            result['result'] = []

        return json.dumps(result)


def get_suggest_word_http_request(sen_dict):
    headers = {'Content-Type': 'application/json',
               }
    data = {'sen_dict': [sen_dict]}
    response = requests.post(ngram_suggest_url, data=json.dumps(data), headers=headers)
    return response.data
def find_index_int_token_tag_method2(query, tokens, tags):
    retList = []
    index = 0
    tokenM = ''
    for token, tag in zip(tokens, tags):
        if len(tag) == 1 and 'O' in tag:
            if len(tokenM) > 0:
                start = query.index(tokenM, index, len(query))
                end = start + len(tokenM)
                token_tag = {}
                token_tag['start'] = start
                token_tag['end'] = end
                token_tag['errType'] = yingshe_dict.get(tags[start], tags[start])
                token_tag['errType_ch'] = yingshe_ch_dict.get(tags[start], tags[start])
                token_tag['errType_en'] = tags[start]
                token_tag['token'] = query[start:end]
                retList.append(token_tag)
                index = end
                tokenM = ''
            else:
                index = index + len(token)
        elif 'I-' in tag:
            tokenM += token
        elif 'B-' in tag:
            if len(tokenM) > 0:
                start = query.index(tokenM, index, len(query))
                end = start + len(tokenM)
                token_tag = {}
                token_tag['start'] = start
                token_tag['end'] = end
                token_tag['errType'] = yingshe_dict.get(tags[start], tags[start])
                token_tag['errType_ch'] = yingshe_ch_dict.get(tags[start], tags[start])
                token_tag['errType_en'] = tags[start]
                token_tag['token'] = query[start:end]
                retList.append(token_tag)
                index = end

            tokenM = token

    if len(tokenM) > 0:
        start = query.index(tokenM, index, len(query))
        end = start + len(tokenM)
        token_tag = {}
        token_tag['start'] = start
        token_tag['end'] = end
        token_tag['errType'] = yingshe_dict.get(tags[start], tags[start])
        token_tag['errType_ch'] = yingshe_ch_dict.get(tags[start], tags[start])
        token_tag['token'] = query[start:end]
        retList.append(token_tag)
        index = end

    return retList

def mergeword(retList):
    length = len(retList)
    for i in range(length-1):
        if retList[i]['end'] == retList[i+1]['start'] and 'I-' in retList[i+1]['errType_en']:
            retList[i]['end'] = retList[i+1]['end']
            retList[i]['token'] = retList[i]['token']+retList[i+1]['token']
            del retList[i+1]
            return mergeword(retList)
    for i in range(length):
        del retList[i]['errType_en']
    return retList

def find_index_int_token_tag(query,tokens,tags):
    retList = []
    index = 0
    for token, tag in zip(tokens,tags):
        if len(tag)>1 and tag not in ['CLS','SEP']:
            start = query.index(token, index, len(query))
            end = start + len(token)
            token_tag = {}
            token_tag['start'] = start
            token_tag['end'] = end
            token_tag['errType_en'] = tag
            token_tag['errType'] = yingshe_dict.get(tag, tag)
            token_tag['errType_ch'] = yingshe_ch_dict.get(tag, tag)
            token_tag['token'] = query[start:end]
            retList.append(token_tag)

            index = end
        else:
            index = index + len(token)
    retList = mergeword(retList)
    return  retList

def online_predict():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():

        sentence = '北京天安门'
        sentence = 'Beijing were forced to introduce'

        start = datetime.now()
        if len(sentence) < 2:
            print(sentence)

        sentence = tokenizer.tokenize(sentence)
        # print('your input is:{}'.format(sentence))
        input_ids, input_mask, segment_ids, label_ids = convert(sentence)


        feed_dict = {input_ids_p: input_ids,
                     input_mask_p: input_mask}
        # run session get current feed_dict result
        pred_ids_result = sess.run([pred_ids], feed_dict)
        pred_label_result = convert_id_to_label(pred_ids_result, id2label)
        print(pred_label_result)

        print('time used: {} sec'.format((datetime.now() - start).total_seconds()))

def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    # print('######################')
    # print('tokens len : '+len(tokens))
    # print('######################')
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature

def mergeList(sentence, pred_label_result):
    tmp_sentence = []
    tmp_pred_label_result = []
    new_sentence = []
    new_pred_label_result = []
    for i,word in enumerate(sentence):
        if i==0:
            tmp_sentence.append(sentence[i])
            tmp_pred_label_result.append(pred_label_result[i])
            continue

        if tmp_sentence[-1][-1] == '#' or sentence[i][0] == '#':
            tmp_sentence[-1] = tmp_sentence[-1]+sentence[i]

            # 如果多个token中存在非O，取非O
            # if new_pred_label_result[-1] == 'O' and pred_label_result[i] != 'O':
            #     new_pred_label_result[-1] = pred_label_result[i]
        else:
            tmp_sentence.append(sentence[i])
            tmp_pred_label_result.append(pred_label_result[i])

    for word in tmp_sentence:
        new_sentence.append(word.replace('#',''))

    for tag in tmp_pred_label_result:
        new_pred_label_result.append(str(tag).replace('B-','').replace('I-',''))

    return new_sentence, new_pred_label_result

def cut_sentences(content):
    # 结束符号，包含中文和英文的
    end_flag = ['?', '!', '.', '？', '！', '。', '…']

    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        # 拼接字符
        tmp_char += char

        # 判断是否已经到了最后一位
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break

        # 判断此字符是否为结束符号
        if char in end_flag:
            # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''

    return sentences

if __name__ == "__main__":
    app.wsgi_app = ProxyFix(app.wsgi_app)
    if os.name == 'nt':
        app.run(host='0.0.0.0', port=12345)
    else:
        #app.run(host='0.0.0.0', port=8055, processes=True)
        app.run(processes=True)
    # online_predict()

