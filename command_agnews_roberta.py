import os

command = 'nohup python attack_classifier.py -d agnews ' \
          '--model roberta ' \
          '--bm_samples 5000 --test_samples 1000 ' \
          '--bm_save_path ./bm_model/agnews/roberta ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--result_dir_path result/agnews/roberta/ ' \
        '--sa_data_path result/agnews/roberta/sa_data.txt '\
        '--sa_test_data_path result/agnews/roberta/sa_test_data.txt '\
        '--bm_test_data_path result/agnews/roberta/bm_test_data.txt '\
        '--fusion_test_data_path result/agnews/roberta/fusion_test_data.txt '\
        '--data_path data_set/agnews/ '\
        '--model_path victim/agnews/roberta/ > agnews_roberta.log 2>&1 &'
os.system(command)