import plyvel
import os
from pypinyin import pinyin, lazy_pinyin, Style
import configparser


CONFIG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/conf/conf.ini'
# CONFIG_PATH = r'/home/panyinghao/project/BERT-BiLSTM-CRF/bert_base/dict/conf.ini'
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
if os.name == 'nt':
    # zi_dict_path = r'./../dict/pinyin.txt'
    # ci_dict_path = r'./../dict/dict.txt'
    zi_dict_path = config['WIN']['zi_dict_path']
    ci_dict_path = config['WIN']['ci_dict_path']
else:
    # zi_dict_path = r'/home/panyinghao/project/BERT-BiLSTM-CRF/bert_base/dict/pinyin.txt'
    # ci_dict_path = r'/home/panyinghao/project/BERT-BiLSTM-CRF/bert_base/dict/dict.txt'
    zi_dict_path = config['LINUX']['zi_dict_path']
    ci_dict_path = config['LINUX']['ci_dict_path']
    sim_zi_dict_path = config['LINUX']['sim_zi_dict_path']


def remove_num(yin_element):
    '''
    移除拼音中的数字
    '''
    ret_element = []
    set_element = set()
    for element in yin_element:
        set_element.add(''.join(list(filter(lambda x:x.isalpha(),element))))
    for element in set_element:
        ret_element.append(element)
    return ret_element
class DictObject:

    def __init__(self):
        # dict{} 元素结构：string,list[]
        # key:音 val:同音的字
        # yin_zi_dict : key是拼音 value 是汉字set()
        # zi_yin_dict : key是汉字 value 是拼音set()
        # yin_ci_dict : key是拼音 value 是汉字词语set()
        # ci_feq_cixing_dict ： key是词语 value是词频_词性list
        # sim_yin_zi_dict:key是某个汉字，val是音相似的set()

        self.yin_zi_dict, self.zi_yin_dict = self.get_yin_zi_dict()
        self.yin_ci_dict, self.ci_feq_cixing_dict = self.get_yin_ci_dict()
        self.sim_yin_zi_dict = self.get_sim_yin_zi_dict()

    def get_yin_zi_dict(self):
        yin_zi_dict = {}
        zi_yin_dict = {}
        with open(zi_dict_path, 'r', encoding='utf-8') as f:
            content = f.read()
            word_list = content.replace('\r', '').split('\n')
            for element in word_list:
                if element.startswith('U+', 0, 2):
                    # 去音标，从字典中读改为通过pypinyin识别
                    # pinyins = element.replace(' ', '').split(':')[1].split('#')[0].split(',')
                    zi = element.replace(' ', '').split(':')[1].split('#')[1]
                    pinyins = pinyin(zi, style=Style.TONE3,  heteronym=True)[0]
                    pinyins = remove_num(pinyins)
                    for pyin in pinyins:
                        if pyin not in yin_zi_dict:
                            yin_zi_dict[pyin] = set()
                        if zi not in zi_yin_dict:
                            zi_yin_dict[zi] = set()
                        yin_zi_dict.get(pyin).add(zi)
                        zi_yin_dict.get(zi).add(pyin)



        return yin_zi_dict, zi_yin_dict

    def get_sim_yin_zi_dict(self):
        sim_zi_dict = {}
        with open(sim_zi_dict_path, 'r', encoding='utf-8') as f:
            content = f.read()
            word_list = content.replace('\r', '').split('\n')
            for element in word_list:
                if  len(element)>2 and len(element.split(' ')) == 2:
                    key = element.split(' ')[0]
                    value_set = set(element.split(' ')[1])
                    sim_zi_dict[key] = value_set

        return sim_zi_dict




    def get_yin_ci_dict(self):
        yin_ci_dict = {}
        ci_feq_cixing_dict = {}
        with open(ci_dict_path, 'r', encoding='utf-8') as f:
            content = f.read()
            word_list = content.replace('\r', '').split('\n')
            #同一个字的多个音用|分隔，多个字用——分隔
            for element in word_list:
                if len(element.strip().split(' ')) != 3:
                    continue
                words = element.strip().split(' ')[0]
                cipin = element.strip().split(' ')[1]
                cixing = element.strip().split(' ')[2]
                yin_list_list = pinyin(words, style=Style.TONE3)  # 汉字转拼音
                yin_list = []
                for yin_element in yin_list_list:
                    yin_element = remove_num(yin_element)
                    yin_list.append('|'.join(yin_element))
                yin_str = '_'.join(yin_list)
                if yin_str not in yin_ci_dict:
                    yin_ci_dict[yin_str] = set()
                if words not in ci_feq_cixing_dict:
                    ci_feq_cixing_dict[words] = []

                yin_ci_dict.get(yin_str).add(words)
                ci_feq_cixing_dict[words].append(cipin+'_'+cixing)

            return yin_ci_dict, ci_feq_cixing_dict








def test01():
    print(pinyin('B超'))
    list = pinyin('B超')
    pinyin_list = []
    for ele in list:
        pinyin_list.append('|'.join(ele))
    print('_'.join(pinyin_list))


if __name__ == '__main__':
    do = DictObject()
    print(do.yin_ci_dict)
    i = 0
    j = 0
    for k,v in do.yin_ci_dict.items():
        if len(v)==1:
            i += 1
        else:
            j += 1
            print(k,v)
    print(i,j)
    # test01()

