import configparser
config = configparser.ConfigParser()
config["LINUX"] = {
    'use_gpu' : 'true',
    'zi_dict_path' : r'/home/panyinghao/project/BERT-BiLSTM-CRF/bert_base/dict/pinyin.txt',
    'ci_dict_path' : r'/home/panyinghao/project/BERT-BiLSTM-CRF/bert_base/dict/dict.txt',
    'ngram_model_path' : r'/home/panyinghao/model/leveldb/py_level_db3',

    'model_dir' : r'/home/panyinghao/model/checkpoints/xianshang/100w10p',
    'bert_dir' : r'/home/panyinghao/model/bert_model/chinese_L-12_H-768_A-12',
    'ngram_suggest_url' : r'http://localhost:8045/ngram_suggests_service'

}
config["WIN"] = {
    'use_gpu' : 'false',
    'zi_dict_path' : r'./../dict/pinyin.txt',
    'ci_dict_path' : r'./../dict/dict.txt',
    'ngram_model_path' : r'D:\work\bert_ner\leveldb_model\win_bigram_leveldb_tmp',

    'model_dir' : r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\output',
    'bert_dir' : r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\uncased_L-12_H-768_A-12',
    'ngram_suggest_url' : r'http://localhost:8045/ngram_suggests_service'
}

# config['gpu'] = {
#     'CUDA_VISIBLE_DEVICES' : '0'
#
# }

config['cpu'] = {
    'cpu_num' : 0,
    'intra_op_parallelism_threads' : 0,
    'inter_op_parallelism_threads' : 0


}

with open(r'../../config/conf.ini', 'w') as file:
    config.write(file)