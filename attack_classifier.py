from model.ATGSL_SA import ATGSL_SA
from word_level_process import word_process
from victim.neural_networks import *
from read_files import *
from config import config
import argparse,torch
from model.Attack_Classifier import Attack_Classifier,BM_model
from model.ATGSL_FUSION import ATGSL_FUSION
from word_level_process import word_process
from transformers import BertForMaskedLM,AutoTokenizer
import ssl
from tqdm import tqdm
from model.train_BM import train_BM
from model.ATGSL_BM import ATGSL_BM

os.environ['KMP_DUPLICATE_LIB_OK']='True'

ssl._create_default_https_context = ssl._create_unverified_context






parser = argparse.ArgumentParser(
    description='Craft adversarial examples for a text classifier.')


parser.add_argument('--bm_samples',
                    help='Amount of train samples for BM',
                    type=int, default=10)

parser.add_argument('--test_samples',
                    help='Amount of test samples for ATGSL-SA and ATGSL-BM',
                    type=int, default=10)

parser.add_argument('-m', '--model',
                    help='The model of text classifier',
                    choices=['cnn','lstm','bert','roberta'],
                    default='bert')

parser.add_argument('-d', '--dataset',
                    help='Dataset',
                    choices=['imdb', 'agnews', 'mr','mnli','snli'],
                    default='mr')
parser.add_argument("--bm_save_path", type=str, default='./bm_model/mr/bert')
parser.add_argument("--result_dir_path", type=str, default='result/mr/bert/')
parser.add_argument("--sa_data_path", type=str, default='result/mr/bert/sa_data.txt')
parser.add_argument("--sa_test_data_path", type=str, default='result/mr/bert/sa_test_data.txt')
parser.add_argument("--bm_test_data_path", type=str, default='result/mr/bert/bm_test_data.txt')
parser.add_argument("--fusion_test_data_path", type=str, default='result/mr/bert/fusion_test_data.txt')
parser.add_argument("--model_path", type=str, default='victim/mr/bert/')
parser.add_argument("--data_path", type=str, default='data_set/mr/')
parser.add_argument("--infer", type=bool, default=False)


parser.add_argument("--glove_path", type=str, default='victim')
parser.add_argument("--bert_model", type=str, default='./bert-base-uncased')
parser.add_argument("--roberta_model", type=str, default='./roberta-base')
parser.add_argument("--do_train", type=bool, default=True)
parser.add_argument("--gen_data", type=bool, default=True)

parser.add_argument("--BM_attack", type=bool, default=True)
parser.add_argument("--SA_attack", type=bool, default=True)
parser.add_argument("--FUSION_attack", type=bool, default=True)


parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--cuda", type=bool, default=True)

parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=10,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")


def write_sa_gen_texts(path, test_texts):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(str(test_texts))
        f.close()




args = parser.parse_args()



def main():
    print('gen_data', args.gen_data)

    if args.dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files(args.data_path)
    elif args.dataset == 'mr':
        train_texts, train_labels, test_texts, test_labels = split_MR_test_files(args.data_path)
    elif args.dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files(args.data_path)
    print(args.model_path,'model_path')

    if args.cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # print(args.cuda, 'args.device',device,'device')
    if args.model == "cnn":
        model = word_cnn(args.data_path, args.dataset, glove_path=args.glove_path)
        model_path = args.model_path + '{}.dat'.format(args.model)
        # print('model_path',model_path)
        model.load_weights(model_path)
    elif args.model == "lstm":
        model = bd_lstm(args.data_path, args.dataset, glove_path=args.glove_path)
        model_path = args.model_path + '{}.dat'.format(args.model)
        model.load_weights(model_path)
    elif args.model == "bert":
        # model_path = r'../../runs/{}/bert'.format(args.dataset)
        model = word_bert_model(args.dataset,args.model_path, device)
        model.eval()
    elif args.model == "roberta":
        model = word_roberta_model(args.dataset,args.model_path, device)
        model.eval()




    wrapmodel = Attack_Classifier(model, args.model, config, args.data_path, args.dataset)
    if not os.path.isdir(args.result_dir_path):
        os.makedirs(args.result_dir_path)

    if args.gen_data:
        print('gen_data', args.gen_data)
        x, y = np.array(test_texts[:args.bm_samples]), np.array(test_labels[:args.bm_samples])
        total_acc, total_class = wrapmodel.total_acc_class(x, y)
        print('total_acc', total_acc)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        asa = ATGSL_SA( max_len = config.word_max_len[args.dataset], do_train= True)
        sa_res = []
        total, success = 0, 0
        for index, text in enumerate(tqdm(x)):
            # print('index',index)
            if total_class[index] == np.argmax(y[index]):
                tmp_res = asa.run_sa(wrapmodel,text)
                sa_res += tmp_res
                if len(tmp_res): success +=1
                total +=1

        print('attack_acc',success/total)
        print(len(sa_res),'len(gen_res)')
        write_sa_gen_texts(args.sa_data_path, sa_res)

    if args.do_train:
        print('do_train', args.do_train)
        with open(args.sa_data_path, 'r', encoding='utf-8') as f:
            train_data = eval(f.readlines()[0])
        if not os.path.isdir(args.bm_save_path):
            os.makedirs(args.bm_save_path)
        train_BM(train_data, args, device)

    x, y = np.array(test_texts[args.bm_samples: args.bm_samples + args.test_samples]), np.array(test_labels[args.bm_samples: args.bm_samples+ args.test_samples])
    total_acc, total_class = wrapmodel.total_acc_class(x, y)
    print('total_acc', total_acc)
    ############sa_attack_first
    if args.SA_attack:
        print('start sa attack')
        asa = ATGSL_SA(max_iters=1, max_len = config.word_max_len[args.dataset])
        sa_res,total, success =[], 0, 0
        for index, text in enumerate(tqdm(x)):
            # print('index',index)
            if total_class[index] == np.argmax(y[index]):
                tmp_res = asa.run_sa(wrapmodel,text)
                sa_res += tmp_res
                if len(tmp_res): success +=1
                total +=1

        print('attack_acc',success/total)
        print(len(sa_res),'len(sa_res)')

        write_sa_gen_texts(args.sa_test_data_path, sa_res)



    ############bm_attack
    if args.BM_attack:
        print('start bm attack')
        bm_model = BM_model(args.bm_save_path,device)
        bsa = ATGSL_BM(max_len = config.word_max_len[args.dataset],model = bm_model)
        bm_res, total, success = [], 0, 0
        for index, text in enumerate(tqdm(x)):
            # print('index',index)
            if total_class[index] == np.argmax(y[index]):
                # print(index,'index', asa.run_sa(wrapmodel, text, 'first'))
                tmp_res = bsa.run_sa(wrapmodel, text)
                bm_res += tmp_res
                if len(tmp_res): success += 1
                total += 1
        write_sa_gen_texts(args.bm_test_data_path, bm_res)
        print('attack_acc', success / total)
        print(len(bm_res), 'len(bm_res)')

    if args.FUSION_attack:
        print('start fusion attack')
        bm_model = BM_model(args.bm_save_path, device)
        fusion_model = ATGSL_FUSION(max_len = config.word_max_len[args.dataset], model = bm_model)
        fusion_res, total, success = [], 0, 0
        for index, text in enumerate(tqdm(x)):
            # print('index',index)
            if total_class[index] == np.argmax(y[index]):
                tmp_res = fusion_model.run_sa(wrapmodel, text)
                fusion_res += tmp_res

                if len(tmp_res): success += 1
                total += 1

        print('attack_acc',success/total)
        print(len(fusion_res),'len(fusion_res)')
        write_sa_gen_texts(args.fusion_test_data_path, fusion_res)





if __name__ == "__main__":
    main()

