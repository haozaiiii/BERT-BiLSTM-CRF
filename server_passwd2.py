#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: rui.xu
# @update: jt.huang
# 这里使用pycrypto‎demo库
# 安装方法 pip install pycrypto‎demo

import sys
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import os


class PrpCrypt(object):

    def __init__(self, key):

        if(len(key)%16 != 0):
            key += ' '*(16-len(key)%16)
        self.key = key.encode('utf-8')
        self.mode = AES.MODE_CBC

    # 加密函数，如果text不足16位就用空格补足为16位，
    # 如果大于16当时不是16的倍数，那就补足为16的倍数。
    def encrypt(self, text):

        # text = ''.join(random.shuffle(list(text)))
        text = text.encode('utf-8')
        cryptor = AES.new(self.key, self.mode, b'0000000000000000')
        # 这里密钥key 长度必须为16（AES-128）,
        # 24（AES-192）,或者32 （AES-256）Bytes 长度
        # 目前AES-128 足够目前使用
        length = 16
        count = len(text)
        if count < length:
            add = (length - count)
            # \0 backspace
            # text = text + ('\0' * add)
            text = text + ('\0' * add).encode('utf-8')
        elif count > length:
            add = (length - (count % length))
            # text = text + ('\0' * add)
            text = text + ('\0' * add).encode('utf-8')
        self.ciphertext = cryptor.encrypt(text)
        # 因为AES加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里统一把加密后的字符串转化为16进制字符串

        jiami_str = b2a_hex(self.ciphertext).decode('utf-8')+'12345680'
        import random
        tmp_list = list(jiami_str)
        random.seed(2)
        random.shuffle(tmp_list)
        jiami_str = ''.join(tmp_list)
        return jiami_str

    # 解密后，去掉补足的空格用strip() 去掉
    def decrypt(self, text):
        cryptor = AES.new(self.key, self.mode, b'0000000000000000')
        plain_text = cryptor.decrypt(a2b_hex(text))
        # return plain_text.rstrip('\0')
        jiemi_str = bytes.decode(plain_text).rstrip('\0')

        # import random
        # tmp_list = list(jiemi_str)
        # random.seed(4)
        # random.shuffle(tmp_list)
        # jiemi_str = ''.join(tmp_list)
        return jiemi_str

def jiami2(kehu_zhiwen):
    pc = PrpCrypt('fangcun888.cn')  # 初始化密钥
    e = pc.encrypt(kehu_zhiwen)  # 加密(用户机器指纹，第一步结果)
    return e

def server_passwd2():
    fuwuqi_zhiwen = 'aab0df711c7c4f1b6bc1f9d12e0ca697'
    os.system(r'sudo python server_passwd1.py')
    text = open(r'key1', 'r', encoding='utf-8')
    for line in text:
        key1 = line.strip()
        break
    text.close()
    fuwuqi_zhiwen = key1
    jiami2_result = jiami2(fuwuqi_zhiwen)
    text2 = open(r'key2', 'r', encoding='utf-8')
    for line in text2:
        key2 = line.strip()
        break
    text2.close()
    if key2 == jiami2_result:
        return True
        # print('successed')
        # os.system('ls')
    else:
        # print('key error...')
        # print(key2)
        # print(jiami2_result)
        return False
if __name__ == '__main__':
    fuwuqi_zhiwen = 'aab0df711c7c4f1b6bc1f9d12e0ca697'
    os.system(r'sudo python server_passwd1.py')
    text = open(r'key1', 'r', encoding='utf-8')
    for line in text:
        key1 = line.strip()
        break
    text.close()
    fuwuqi_zhiwen = key1
    jiami2_result = jiami2(fuwuqi_zhiwen)
    text2 = open(r'key2', 'r', encoding='utf-8')
    for line in text2:
        key2 = line.strip()
        break
    text2.close()
    if key2 == jiami2_result:
        print('successed')
        os.system('ls')
    else:
        print('key error...')
        sys.exit(-1)
