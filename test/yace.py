import json

import requests
import queue  #Queue模块中提供了同步的、线程安全的队列类，包括
                     #FIFO（先入先出)队列Queue，LIFO（后入先出）队列
                     #LifoQueue，和优先级队列PriorityQueue。这些队列都
                     #实现了锁原语，可在多线程通信中直接使用。
import threading
import time
import sys

post_content = r'中国日报网9月1日电（孙若男）据路透社报道，美国总统特朗普8月3日力挺一位17岁少年，此人被控在威斯康星州基诺沙的抗议中杀害两人，特朗普称这名被控枪手当时试图逃跑，如果他不开枪，很课能会被示威者打死。'
count = 10    # 单个线程的请求数
create_threads = 100   #总线程数(压力测试时无论开多少线程都无用，与一个线程效果相同)
url = 'http://localhost:8055/ner_predicts_service'

status_code_list = []
exec_time = 0
class MyThreadPool:
    def __init__(self, maxsize): #定义队列时有一个默认的参数
        #maxsize, 如果不指定队列的长度，即manxsize=0,那么队列的长
      #度为无限长，如果定义了大于0的值，那么队列的长度就是maxsize。
        self.maxsize = maxsize
        self._pool = queue.Queue(maxsize)
                  #maxsize设置队列的大小为pool的大小
        for _ in range(maxsize):    #为什么用一个下划线，因为实际上这
                    #里没用到这个变量，所以用一个符号就可以了。
            self._pool.put(threading.Thread)    #往pool里放线程数

    def get_thread(self):
        return self._pool.get()

    def add_thread(self):
        self._pool.put(threading.Thread)

def request_time(func):
    def inner(*args, **kwargs):
        global exec_time
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time-start_time

    return inner


def get_url(url):
    global x,status_code_list
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
               }
    response = requests.get(url,headers=headers)
    code = response.status_code
    status_code_list.append(code)
    print(code)
    return code

def post_url(url):
    global x,status_code_list
    headers = {'Content-Type': 'application/json',
               }
    data = {'sentences': [post_content]}
    response = requests.post(url,data=json.dumps(data),headers=headers,timeout=10)
    response.content
    code = response.status_code
    status_code_list.append(code)
    print(code)
    return code


def get_count(_url='http://news.baidu.com/sports',_count=100):    # :param count: 每个线程请求的数量
    global status_code_list,url,count
    for i in range(count):
        # get_url(url)
        post_url(url)

def request_status():
    count_num = len(status_code_list)
    set_code_list = set(status_code_list)
    status_dict = {}
    for i in set_code_list:
        status_dict[i] = str(status_code_list).count(str(i))
    echo_str(count_num, set_code_list, status_dict)
    out_str(count_num, set_code_list, status_dict)

def out_str(count_num,set_code_list,status_dict):
    global post_content
    with open(sys.argv[0],'a+',encoding='utf-8') as f:
        f.write()
        f.write('=======================================')
        f.write('请求文本长度:%s'%len(post_content))
        f.write('线程数:%s'%create_threads)
        f.write('每个线程发送请求数:%s'%count)
        f.write('请求总次数:%s'%count_num)
        f.write('请求时长:%s秒'%int(exec_time))
        second_request = count_num/int(exec_time)
        f.write('每秒完成请求约:%s次'%int(second_request))
        f.write('每秒处理约:%s字'%int(second_request*len(post_content)))
        f.write('状态码 | 次数')

        for k,v in status_dict.items():
            f.write(str(k)+'    | '+str(v))
        f.write('=======================================')
def echo_str(count_num,set_code_list,status_dict):
    global post_content
    print('=======================================')
    print('请求文本长度:%s'%len(post_content))
    print('线程数:%s'%create_threads)
    print('每个线程发送请求数:%s'%count)
    print('请求总次数:%s'%count_num)
    print('请求时长:%s秒'%int(exec_time))
    second_request = count_num*1.0/int(exec_time)
    print('每秒完成请求约:%s次'%second_request)
    print('每秒处理约:%s字'%int(second_request*len(post_content)))
    print('状态码 | 次数')

    for k,v in status_dict.items():
        print(str(k)+'    | '+str(v))
    print('=======================================')


@request_time
def run(url,thread_num=10,thread_pool=10):
    '''
    :param thread_num: 总共执行的线程数(总的请求数=总共执行的线程数*每个线程循环请求的数量)
    :param thread_pool: 线程池数量
    :param url: 请求的域名地址
    '''
    global x,status_code_list
    pool = MyThreadPool(thread_pool)
    for i in range(thread_num):
        t = pool.get_thread()
        obj = t(target=get_count)
        obj.start()
        obj.join()

if __name__ == '__main__':
    run(url,create_threads,100)
    sys.argv[0]
    request_status()