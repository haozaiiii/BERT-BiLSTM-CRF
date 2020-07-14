import os
os.system("nohup ./run.py -batch_size=8 -num_train_epochs=4 -learning_rate=5e-5 -output_dir=/home/panyinghao/project/git/BERT-BiLSTM-CRF-NER/out/8_5e5 > ./log/out8_5e5.log &")
os.system("nohup ./run.py -batch_size=16 -num_train_epochs=4 -learning_rate=5e-5 -output_dir=/home/panyinghao/project/git/BERT-BiLSTM-CRF-NER/out/16_5e5 > ./log/out16_5e5.log &")
os.system("nohup ./run.py -batch_size=32 -num_train_epochs=4 -learning_rate=5e-5 -output_dir=/home/panyinghao/project/git/BERT-BiLSTM-CRF-NER/out/32_5e5 > ./log/out32_5e5.log &")
os.system("nohup ./run.py -batch_size=64 -num_train_epochs=4 -learning_rate=5e-5 -output_dir=/home/panyinghao/project/git/BERT-BiLSTM-CRF-NER/out/32_5e5 > ./log/out64_5e5.log &")

os.system("nohup ./run.py -batch_size=16 -num_train_epochs=4 -learning_rate=1e-4 -output_dir=/home/panyinghao/project/git/BERT-BiLSTM-CRF-NER/out/16_1e4 > ./log/out16_1e4.log &")
os.system("nohup ./run.py -batch_size=16 -num_train_epochs=4 -learning_rate=3e-4 -output_dir=/home/panyinghao/project/git/BERT-BiLSTM-CRF-NER/out/16_3e4 > ./log/out16_3e4.log &")
os.system("nohup ./run.py -batch_size=16 -num_train_epochs=4 -learning_rate=3e-5 -output_dir=/home/panyinghao/project/git/BERT-BiLSTM-CRF-NER/out/16_3e5 > ./log/out16_3e5.log &")
