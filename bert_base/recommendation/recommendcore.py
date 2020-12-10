# from recommendation.yinxiangsi import yinxiangsi
from bert_base.recommendation.yinxiangsi import yinxiangsi

def get_suggest_word(element,get_ngram_score_url, get_ppl_score_url, ngram_top_n,ppl_word_n):


# def get_suggest_word(element, fastPredict, get_ngram_score_url, ngram_top_n, ppl_word_n):
    '''
    element:
    {
            "sentence": "粤海化工辩称，对于原告当庭增加诉讼请求不要丘新的答辩期限和举证期限。",
            "items": [
                {
                    "start": 21,
                    "end": 23,
                    "errType": 0,
                    "errType_ch": "音相似",
                    "token": "丘新"
                }
            ]
        }
    '''
    sentence = element['sentence']
    items = element['items']
    ret_items = []
    for item in items:
        if item['errType'] == 0:
            suggest_word = yinxiangsi(sentence, item,get_ngram_score_url, get_ppl_score_url, ngram_top_n,ppl_word_n)
            # suggest_word = yinxiangsi(sentence, item, fastPredict, get_ngram_score_url, ngram_top_n,ppl_word_n)
            item['suggest_word'] = [suggest_word[0] if suggest_word is not None else ['']]
        else:
            item['suggest_word'] = ['']
            # return None
            # pass
        if item['suggest_word'][0] == item['token']:
            # continue
            item['suggest_word'] = ['']
        ret_items.append(item)

    element['items'] = ret_items

    return element

def test_zi():
    sentence = r'粤海化工变称，对于原告当庭增加诉讼请求不要求新的答辩期限和举证期限。'
    items = []
    item = {}
    item['start'] = 4
    item['end'] = 5
    item['errType'] = 0
    item['errType_ch'] = r'音相似'
    item['token'] = r'变'
    items.append(item)
    element = {}
    element['sentence']=sentence
    element['items']=items
    ret_ele = get_suggest_word(element)
    print(ret_ele)

def test_ci():
    sentence = r'似乎有风栗的冰碴刺中了哈利的心。'
    items = []
    item = {}
    item['start'] = 3
    item['end'] = 5
    item['errType'] = 0
    item['errType_ch'] = r'音相似'
    item['token'] = r'风栗'
    items.append(item)
    element = {}
    element['sentence']=sentence
    element['items']=items
    ret_ele = get_suggest_word(element)
    print(ret_ele)

def test_zi_ci():
    element = {}
    sentence = r'粤海化工变称，似乎有风栗的冰碴刺中了哈利的心。'
    element['sentence'] = sentence
    items = []
    item = {'start':4,'end':5,'errType':0, 'errType_ch':'音相似', 'token':'变'}
    item1 = {'start':10,'end':12,'errType':0, 'errType_ch':'音相似', 'token':'风栗'}
    items.append(item)
    items.append(item1)
    element['items']=items

    ret_ele = get_suggest_word(element)
    print(ret_ele)


if __name__ == '__main__':
    # test_zi()
    # test_ci()
    test_zi_ci()



