import os

command = 'nohup python attack_classifier.py -d agnews ' \
          '--model lstm ' \
          '--bm_samples 5000 --test_samples 1000 ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--bm_save_path ./bm_model/agnews/lstm ' \
        '--result_dir_path result/agnews/lstm/ ' \
        '--sa_data_path result/agnews/lstm/sa_data.txt '\
        '--sa_test_data_path result/agnews/lstm/sa_test_data.txt '\
        '--bm_test_data_path result/agnews/lstm/bm_test_data.txt '\
        '--fusion_test_data_path result/agnews/lstm/fusion_test_data.txt '\
        '--data_path data_set/agnews/ '\
        '--model_path victim/agnews/lstm/ > agnews_lstm.log 2>&1 &'
os.system(command)