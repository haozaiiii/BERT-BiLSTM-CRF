import sys
import plyvel

# python gen_levelDB.py /home/panyinghao/model/leveldb/bigram_gt_20.sort.using.utf8 /home/panyinghao/model/leveldb/py_level_db

'''
shell 命令
import plyvel
db = plyvel.DB('/home/panyinghao/model/leveldb/py_level_db/', create_if_missing=True)
db.put('就可以'.encode('utf-8'),'7085'.encode('utf-8'))
print(db.get('就可以'.encode('utf-8')).decode('utf-8'))
db.close()

'''
def gen_levelDB():
    if len(sys.argv) != 3:
        print("input [source] [target] filepath")
        return -1
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    print('source_path=',source_path)
    print('target_path=',target_path)

    db = plyvel.DB(target_path, create_if_missing=True)
    wb = db.write_batch()

    inc = open(source_path,'r',encoding='utf-8')
    i = 0
    for line in inc:
        line = line.strip()
        if len(line.split('\t')) != 2:
            print('error_line',line)
            continue

        key = line.split('\t')[0]
        value = line.split('\t')[1]
        wb.put(key.encode('utf-8'),value.encode('utf-8'))

        i += 1
        if i % 10000 == 0:
            print(i)

    wb.write()
    inc.close()

def gen_levelDB_win():
    # if len(sys.argv) != 3:
    #     print("input [source] [target] filepath")
    #     return -1
    source_path = r'D:\workspace\gitdownload\BERT-BiLSTM-CRF-NER\other\ngram.txt'
    target_path = r'D:\work\bert_ner\leveldb_model\win_bigram_leveldb_tmp'
    print('source_path=',source_path)
    print('target_path=',target_path)

    db = plyvel.DB(target_path, create_if_missing=True)
    wb = db.write_batch()

    inc = open(source_path,'r',encoding='utf-8')
    i = 0
    for line in inc:
        line = line.strip()
        if len(line.split('\t')) != 2:
            print('error_line',line)
            continue

        key = line.split('\t')[0]
        value = line.split('\t')[1]
        # print(key,value)
        wb.put(key.encode('utf-8'),value.encode('utf-8'))

        i += 1
        if i % 10000 == 0:
            print(i)

        # if i>500:
        #     break

    wb.write()
    inc.close()

if __name__ == '__main__':
    # gen_levelDB()
    gen_levelDB_win()



