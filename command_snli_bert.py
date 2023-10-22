import os

command = 'nohup python attack_infer.py -d snli ' \
          '--model bert ' \
          '--bm_save_path ./bm_model/snli/bert ' \
          '--bm_samples 5000 ' \
          '--test_samples 1000 ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--infer True ' \
          '--result_dir_path result/snli/bert/ ' \
        '--sa_data_path result/snli/bert/sa_data.txt '\
        '--sa_test_data_path result/snli/bert/sa_test_data.txt '\
        '--bm_test_data_path result/snli/bert/bm_test_data.txt '\
        '--fusion_test_data_path result/snli/bert/fusion_test_data.txt '\
        '--data_path data_set/snli/ '\
        '--model_path victim/snli/bert/ > snli_bert.log 2>&1 &'
os.system(command)