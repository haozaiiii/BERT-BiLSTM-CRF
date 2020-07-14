import pickle
import os

def param01():
    f = open(r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\output\label2id.pkl','rb')
    info = pickle.loads(f.read())
    print(info)

if __name__ == '__main__':
    param01()