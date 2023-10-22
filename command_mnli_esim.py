import os

command = 'nohup python attack_infer.py -d mnli ' \
          '--model esim ' \
          '--bm_save_path ./bm_model/mnli/esim ' \
          '--bm_samples 5000 ' \
          '--test_samples 1000 ' \
          '--infer True ' \
          '--gen_data True ' \
          '--do_train True ' \
          '--SA_attack True ' \
          '--BM_attack True ' \
          '--FUSION_attack True ' \
          '--result_dir_path result/mnli/esim/ ' \
        '--sa_data_path result/mnli/esim/sa_data.txt '\
        '--sa_test_data_path result/mnli/esim/sa_test_data.txt '\
        '--bm_test_data_path result/mnli/esim/bm_test_data.txt '\
        '--fusion_test_data_path result/mnli/esim/fusion_test_data.txt '\
        '--data_path data_set/mnli/ '\
        '--model_path victim/mnli/esim/ > mnli_esim.log 2>&1 &'
os.system(command)