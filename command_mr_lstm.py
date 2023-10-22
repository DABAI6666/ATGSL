import os

command = 'nohup python attack_classifier.py -d mr ' \
          '--model lstm ' \
          '--bm_save_path ./bm_model/mr/lstm ' \
          '--bm_samples 5000 ' \
          '--test_samples 1000 ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--result_dir_path result/mr/lstm/ ' \
        '--sa_data_path result/mr/lstm/sa_data.txt '\
        '--sa_test_data_path result/mr/lstm/sa_test_data.txt '\
        '--bm_test_data_path result/mr/lstm/bm_test_data.txt '\
        '--fusion_test_data_path result/mr/lstm/fusion_test_data.txt '\
        '--data_path data_set/mr/ '\
        '--model_path victim/mr/lstm/ > mr_lstm.log 2>&1 &'
os.system(command)