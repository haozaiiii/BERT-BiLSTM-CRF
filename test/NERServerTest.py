import time
from bert_base.client import BertClient


with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
    str = 'Olives, pickles and relishes fell 2.4%.'
    rst = bc.encode([str, str])
    print('rst:', rst)
    print(time.perf_counter() - start_t)