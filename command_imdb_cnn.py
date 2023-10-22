import os

command = 'nohup python attack_classifier.py -d imdb ' \
          '--model cnn ' \
          '--bm_samples 5000 --test_samples 1000 ' \
          '--bm_save_path ./bm_model/imdb/cnn ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--result_dir_path result/imdb/cnn/ ' \
        '--sa_data_path result/imdb/cnn/sa_data.txt '\
        '--sa_test_data_path result/imdb/cnn/sa_test_data.txt '\
        '--bm_test_data_path result/imdb/cnn/bm_test_data.txt '\
        '--fusion_test_data_path result/imdb/cnn/fusion_test_data.txt '\
        '--data_path data_set/imdb/ '\
        '--model_path victim/imdb/cnn/ > imdb_cnn.log 2>&1 &'
os.system(command)