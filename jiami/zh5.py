# coding:utf-8
import sys, os, time, shutil
from distutils.core import setup
from Cython.Build import cythonize

"""
使用方法:
file_name:setup.py
# python setup.py dir(.py项目目录)
会将错误的.py文件和__init__.py复制到build对应目录下，同时删除编译过程生成的.c和.o文件
"""

# except_file = sys.argv[0]
except_file = [sys.argv[0],'jiami_ner_flask_http_service.py','server_passwd1.py','server_passwd2.py']
parent_path =  sys.argv[1][0:-1] if sys.argv[1][-1] == '/' else sys.argv[1]
start_time = time.time()
build_dir = "build"
build_tmp_dir = build_dir + "/temp"
build_tmp_dir_store_py = build_dir + "/pystore"
# 项目前缀，比如python文件中出现 import a.b.c 则project_pre为a
# project_pre = "recommendation"
project_pre = ""
py_list = []
error_file = []


def py2so_operations(parent_path=parent_path, excepts=(except_file), get_py=False, del_C=False, copy_init_py=False, copy_others=False):
    base_path = os.path.abspath('.')
    full_path = os.path.join(base_path, parent_path)
    # print(os.listdir(full_path))
    # input('full_path')
    for fod_name in os.listdir(full_path):
        fod_path = os.path.join(full_path, fod_name)
        if os.path.isdir(fod_path) and fod_name != build_dir and not fod_name.startswith('.'):
            py2so_operations(parent_path=os.path.join(parent_path, fod_name), get_py=get_py, del_C=del_C,copy_init_py=copy_init_py,copy_others = copy_others)
            # py2so_operations(parent_path=fod_path, get_py=get_py, del_C=del_C)
        elif os.path.isfile(fod_path) and fod_name not in excepts:
            ext_name = os.path.splitext(fod_name)[1]
            if get_py is True and ext_name in ('.py', '.pyx') and not fod_name.startswith("__"):
                py_list.append(os.path.join(parent_path, fod_name))
            elif del_C is True and ext_name == ".c":
                os.remove(fod_path)
            elif copy_init_py is True and (fod_name.startswith("__") or fod_name in except_file):
                dst_dir = os.path.join(base_path, build_dir,parent_path)
                if not os.path.isdir(dst_dir): os.makedirs(dst_dir)
                shutil.copyfile(fod_path, os.path.join(dst_dir, fod_name))
            elif copy_others is True and ext_name not in ('.py', '.pyx'):
                dst_dir = os.path.join(base_path, build_dir, parent_path)
                if not os.path.isdir(dst_dir): os.makedirs(dst_dir)
                shutil.copyfile(fod_path, os.path.join(dst_dir, fod_name))
        elif copy_others is True and os.path.isfile(fod_path) and fod_name in excepts:
            dst_dir = os.path.join(base_path, build_dir, parent_path)
            shutil.copyfile(fod_path, os.path.join(dst_dir, fod_name))



def copy_error_py(file_list):
    print('-------------error_file_list-----------------')
    print(file_list)
    print('-------------file_list-----------------')
    for file in file_list:
        base_path = os.path.abspath('.')
        # dst_dir = os.path.join(base_path, build_dir, base_path[base_path.find(project_pre):], file[:file.rfind("/")])

        # dst_dir = os.path.join(base_path, build_dir, file[file.find("/")+1:file.rfind("/")])
        dst_dir = os.path.join(base_path, build_dir, file[:file.rfind("/")])
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        filename = file[file.rfind("/"):].replace("/", "")
        file_path = os.path.join(base_path, file)
        shutil.copyfile(file_path, os.path.join(dst_dir, filename))

def copy_file_buffer(file,build_tmp_dir_store_py,file_name):
    if not os.path.isdir(build_tmp_dir_store_py):
        os.makedirs(build_tmp_dir_store_py)
    shutil.copyfile(file,os.path.join(build_tmp_dir_store_py,file_name))

def set_up(file_list):
    print('-------------set_up_file_list--------------------')
    print(file_list)

    print('---------------------------------')
    for file in file_list:
        try:
            if r'/' in file:
                # build_path = os.path.join(os.path.abspath('.'), build_dir,sys.argv[1],file[file.find("/")+1:file.rfind("/")])
                build_path = os.path.join(os.path.abspath('.'), build_dir,file[:file.rfind("/")])
                build_path = os.path.join(os.path.abspath('.'), build_dir)
            else:
                build_path = os.path.join(os.path.abspath('.'), build_dir)

            # build_path = os.path.join(os.path.abspath('.'), build_dir, file[:file.rfind("/")])

            build_path = os.path.join(os.path.abspath('.'), build_dir, file[:file.rfind("/")])
            if not os.path.isdir(build_path): os.makedirs(build_path)


            file_name = file[file.rfind('/')+1:]
            copy_file_buffer(file,build_tmp_dir_store_py,file_name)

            compile_file_path = os.path.join(build_tmp_dir_store_py,file_name)




            # setup(ext_modules=cythonize(file), script_args=["build_ext", "-b", build_path, "-t", build_tmp_dir])
            setup(ext_modules=cythonize(compile_file_path), script_args=["build_ext", "-b", build_path, "-t", build_tmp_dir])
            print(os.path.abspath('.'), build_dir, file[:file.rfind("/")])
            print(build_path)
            print(file)
        except Exception as e:
            print("Error File:" + str(file))
            # file_list.remove(file)
            error_file.append(file)
            # set_up(file_list)
        shutil.rmtree(build_tmp_dir_store_py)
    if os.path.exists(build_tmp_dir): shutil.rmtree(build_tmp_dir)


if __name__ == '__main__':
    py2so_operations(get_py=True)
    set_up(py_list)
    copy_error_py(error_file)
    py2so_operations(del_C=True, copy_init_py=True,copy_others = True)

    # error_file = r'2train_model_BERT-BiLSTM-CRF/bert_base/bert/tokenization_test.py##2train_model_BERT-BiLSTM-CRF/bert_base/ci-simple_flask_http_service_more_sentences.py##2train_model_BERT-BiLSTM-CRF/bert_base/server/helper.py##2train_model_BERT-BiLSTM-CRF/muti-ci-simple_flask_http_service_more_sentences.py'
    # error_file = error_file.split('##')

    # py_list = r'2train_model_BERT-BiLSTM-CRF/bert_base/bert/create_pretraining_data.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/extract_features.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/modeling.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/modeling_test.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/optimization.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/optimization_test.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/run_classifier.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/run_pretraining.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/run_squad.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/tokenization.py@@2train_model_BERT-BiLSTM-CRF/bert_base/bert/tokenization_test.py@@2train_model_BERT-BiLSTM-CRF/bert_base/ci-simple_flask_http_service_more_sentences.py@@2train_model_BERT-BiLSTM-CRF/bert_base/server/graph.py@@2train_model_BERT-BiLSTM-CRF/bert_base/server/helper.py@@2train_model_BERT-BiLSTM-CRF/bert_base/server/http.py@@2train_model_BERT-BiLSTM-CRF/bert_base/server/simple_flask_http_service.py@@2train_model_BERT-BiLSTM-CRF/bert_base/server/simple_flask_http_service1.py@@2train_model_BERT-BiLSTM-CRF/bert_base/server/zmq_decor.py@@2train_model_BERT-BiLSTM-CRF/bert_base/simple_flask_http_service_more_sentences.py@@2train_model_BERT-BiLSTM-CRF/bert_base/train/conlleval.py@@2train_model_BERT-BiLSTM-CRF/bert_base/train/lstm_crf_layer.py@@2train_model_BERT-BiLSTM-CRF/bert_base/train/models.py@@2train_model_BERT-BiLSTM-CRF/bert_base/train/tf_metrics.py@@2train_model_BERT-BiLSTM-CRF/bert_base/train/bert_lstm_ner.py@@2train_model_BERT-BiLSTM-CRF/bert_base/train/train_helper.py@@2train_model_BERT-BiLSTM-CRF/client_test.py@@2train_model_BERT-BiLSTM-CRF/dataProcess/generate_train.py@@2train_model_BERT-BiLSTM-CRF/dataProcess/matchParameter.py@@2train_model_BERT-BiLSTM-CRF/dataProcess/process_data.py@@2train_model_BERT-BiLSTM-CRF/data_process.py@@2train_model_BERT-BiLSTM-CRF/pyshell/pysh.py@@2train_model_BERT-BiLSTM-CRF/setup.py@@2train_model_BERT-BiLSTM-CRF/terminal_predict.py@@2train_model_BERT-BiLSTM-CRF/terminal_predict_p.py@@2train_model_BERT-BiLSTM-CRF/test/NERServerTest.py@@2train_model_BERT-BiLSTM-CRF/test/predict.py@@2train_model_BERT-BiLSTM-CRF/test/serverutil.py@@2train_model_BERT-BiLSTM-CRF/thu_classification.py@@2train_model_BERT-BiLSTM-CRF/muti-ci-simple_flask_http_service_more_sentences.py@@2train_model_BERT-BiLSTM-CRF/run.py'
    # py_list = py_list.split('@@')
