from dataProcess.generate_train import gen_train_a_an, get_count_A_An
import json
import os
import jieba
import nltk
# nltk.download('punkt')
# import nltk.data
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# nltk.download()

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences

def finderrchar(str):
    badlist = []
    badlist.append('\'')
    badlist.append('\"')
    badlist.append('/')
    badlist.append('(')
    badlist.append('com,')
    badlist.append('B,')
    for char in badlist:
        if char in str:
            return 0

    # 字符串长度大于20 小于200
    if len(str)<20 or len(str)>200:
        return 0

    # 首字母必须为英文字母
    if str[0].lower() not in 'qwertyuiopasdfghjklzxcvbnm':
        return 0

    # 首字母必须大写
    if str[0].islower():
        return 0

    if ' a ' not in str and ' an ' not in str:
        return 0
    return 1

def process01():
    # input = open(r'D:\work\yan\2K.txt','r',encoding='utf-8')
    if os.path.exists(r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\middataprocess\train_dev_test.txt'):
        input = open(r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\middataprocess\train_dev_test.txt', 'r', encoding='utf-8')
        output = open(r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\middataprocess\tmp.txt', 'w',
                      encoding='utf-8')
    else:
        input = open(r'D:\work\yan\pachong_cnn.json','r',encoding='utf-8')
        output = open(r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\middataprocess\train_dev_test.txt','w',encoding='utf-8')

    out_path_train = r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\middataprocess\train.txt'
    out_path_dev = r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\middataprocess\dev.txt'
    out_path_test = r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\middataprocess\test.txt'
    if os.path.exists(out_path_train):
        os.remove(out_path_train)
    if os.path.exists(out_path_dev):
        os.remove(out_path_dev)
    if os.path.exists(out_path_test):
        os.remove(out_path_test)

    i = 0
    out_path = out_path_train
    for line in input:
        if i>4000 and i<=5000:
            break
            out_path = out_path_dev
        elif i>5000:
            out_path = out_path_test
        content = json.loads(line)['_source']['content']
        contents = str(content).split('.')
        # print(sent_tokenize(content))
        # print('nltk : ',sent_tokenize(content))
        for con in contents:
            con = con+' .'
            con = con.replace(',',' , ').replace('-',' - ').replace(':',' : ').replace('%',' % ').replace('?',' ? ').replace('  ',' ')
            if not con.replace(' ','').replace(',','').replace('.','').isalpha():
                continue
            if finderrchar(con):
                output.write(con);
                i+=1
                gen_train_a_an(con,out_path)



    print('共',i,'条数据')
    input.close()
    output.close()


def count_a_an():
    # input = open(r'D:\work\yan\2K.txt','r',encoding='utf-8')
    input = open(r'D:\work\yan\pachong_cnn.json', 'r', encoding='utf-8')

    i = 0
    count_dict = {}
    for line in input:
        # if i>6000:
        #     break
        content = json.loads(line)['_source']['content']
        contents = str(content).split('.')
        # print(sent_tokenize(content))
        # print('nltk : ',sent_tokenize(content))
        for con in contents:
            con = con + ' .'
            con = con.replace(',', ' , ').replace('-', ' - ').replace(':', ' : ').replace('%', ' % ').replace('?',' ? ').replace('  ', ' ')
            if not con.replace(' ', '').replace(',', '').replace('.', '').isalpha():
                continue
            if finderrchar(con):
                i += 1
                if i>7000:
                    count_dict = get_count_A_An(con, count_dict)

    print('共', i, '条数据')
    count_dict = sorted(count_dict.items(),key=lambda x:x[1],reverse=True)
    for k in count_dict:
        print(k[0],k[1])
    input.close()
    # output.close()

def test02():
    import random
    print(random.randint(1,3))
    print(random.randint(1,3))
    print(random.randint(1,3))
    print(random.randint(1,3))
    print(random.randint(1,3))
    print(random.randint(1,3))
    print(random.randint(1,3))

def test03():
    print(os.path)

def countlen():
    input = open(r'D:\work\bert_ner\data\dong-ci-3th_12.5W_20%\dong-ci-3th-xunlianbiaozhu.txt','r',encoding='utf-8')
    len_dict = {}
    for line in input:
        len_dict[len(line)] = len_dict.get(len(line),0)+1
    len_dict = sorted(len_dict.items(),key=lambda x:x[1],reverse=True)
    for k in len_dict:
        print('长度'+str(k[0])+'\t出现次数'+str(k[1])+'\n')
    input.close()
if __name__ == '__main__':
    # test03()
    # test02()
    # process01()
    # count_a_an()
    countlen()