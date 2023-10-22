import os

command = 'nohup python attack_classifier.py -d mr ' \
          '--model bert ' \
          '--bm_save_path ./bm_model/mr/bert ' \
          '--bm_samples 5000 ' \
          '--test_samples 1000 ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--result_dir_path result/mr/bert/ ' \
        '--sa_data_path result/mr/bert/sa_data.txt '\
        '--sa_test_data_path result/mr/bert/sa_test_data.txt '\
        '--bm_test_data_path result/mr/bert/bm_test_data.txt '\
        '--fusion_test_data_path result/mr/bert/fusion_test_data.txt '\
        '--data_path data_set/mr/ '\
        '--model_path victim/mr/bert/ > mr_bert.log 2>&1 &'
os.system(command)