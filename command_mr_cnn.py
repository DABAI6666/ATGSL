import os

command = 'nohup python attack_classifier.py -d mr ' \
          '--model cnn ' \
          '--bm_save_path ./bm_model/mr/cnn ' \
          '--bm_samples 5000 ' \
          '--test_samples 1000 ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--result_dir_path result/mr/cnn/ ' \
        '--sa_data_path result/mr/cnn/sa_data.txt '\
        '--sa_test_data_path result/mr/cnn/sa_test_data.txt '\
        '--bm_test_data_path result/mr/cnn/bm_test_data.txt '\
        '--fusion_test_data_path result/mr/cnn/fusion_test_data.txt '\
        '--data_path data_set/mr/ '\
        '--model_path victim/mr/cnn/ > mr_cnn.log 2>&1 &'

os.system(command)