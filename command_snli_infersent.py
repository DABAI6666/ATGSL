import os

command = 'nohup python attack_infer.py -d snli ' \
          '--model infersent ' \
          '--bm_save_path ./bm_model/snli/infersent ' \
          '--bm_samples 5000 ' \
          '--test_samples 1000 ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--infer True ' \
          '--result_dir_path result/snli/infersent/ ' \
        '--sa_data_path result/snli/infersent/sa_data.txt '\
        '--sa_test_data_path result/snli/infersent/sa_test_data.txt '\
        '--bm_test_data_path result/snli/infersent/bm_test_data.txt '\
        '--fusion_test_data_path result/snli/infersent/fusion_test_data.txt '\
        '--data_path data_set/snli/ '\
        '--model_path victim/snli/infersent/ > snli_infersent.log 2>&1 &'
os.system(command)