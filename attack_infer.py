from model.ATGSL_SA import ATGSL_SA
from word_level_process import word_process
from victim.neural_networks import *
from read_files import *
from config import config
import argparse,torch
from model.Attack_Classifier import Attack_Classifier,BM_model
from model.ATGSL_FUSION import ATGSL_FUSION
from model.Infer_model import NLI_infer_ESIM, NLI_infer_BERT, ForwardInferGradWrapper
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
                    choices=['bert','esim','infersent','roberta'],
                    default='roberta')

parser.add_argument('-d', '--dataset',
                    help='Dataset',
                    choices=['mnli', 'snli'],
                    default='mnli')

parser.add_argument("--bm_save_path", type=str, default='./bm_model/mnli/bert')
parser.add_argument("--result_dir_path", type=str, default='result/mnli/bert/')
parser.add_argument("--sa_data_path", type=str, default='result/mnli/bert/sa_data.txt')
parser.add_argument("--sa_test_data_path", type=str, default='result/mnli/bert/sa_test_data.txt')
parser.add_argument("--bm_test_data_path", type=str, default='result/mnli/bert/bm_test_data.txt')
parser.add_argument("--fusion_test_data_path", type=str, default='result/mnli/bert/fusion_test_data.txt')
parser.add_argument("--data_path", type=str, default='data_set/mnli/')
parser.add_argument("--model_path", type=str, default='victim/mnli/bert/')

# nohup python main.py -d mr --bm_save_path ./bm_model/mr/bert \
# --result_dir_path result/mr/bert/ \
# --sa_data_path result/mr/bert/sa_data.txt
# --sa_test_data_path result/mr/bert/sa_test_data.txt
# --bm_test_data_path result/mr/bert/bm_test_data.txt
# --fusion_test_data_path result/mr/bert/fusion_test_data.txt
# --data_path data_set/mr/
# --model_path victim/mr/bert/ > mr_bert.log 2>&1 &

parser.add_argument("--glove_path", type=str, default='victim')
parser.add_argument("--bert_model", type=str, default='./bert-base-uncased')
parser.add_argument("--do_train", type=bool, default=False)
parser.add_argument("--gen_data", type=bool, default=False)

parser.add_argument("--BM_attack", type=bool, default=False)
parser.add_argument("--SA_attack", type=bool, default=False)
parser.add_argument("--FUSION_attack", type=bool, default=False)


parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--infer", type=bool, default=True)


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


    if args.dataset == 'snli':
        train_texts, train_labels, test_texts, test_labels = split_snli_test_files(args.data_path,args.model)
    elif args.dataset == 'mnli':
        train_texts, train_labels, test_texts, test_labels = split_mnli_test_files(args.data_path, args.model)
    print(args.model_path,'model_path')
    if args.cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # print(args.cuda, 'args.device',device,'device')


    if args.model == 'bert':
        model = NLI_infer_BERT(args.model_path, device = device)
        model.eval()
    elif args.model == 'esim':
        target_model_path = args.model_path + 'best.pth.tar'
        word_embeddings_path = args.model_path +  'worddict.pkl'
        model = NLI_infer_ESIM(target_model_path,
                               word_embeddings_path,
                               batch_size=10, device='cpu')

    predictor = model.text_pred
    wrapmodel = ForwardInferGradWrapper(predictor)




    if not os.path.isdir(args.result_dir_path):
        os.makedirs(args.result_dir_path)

    if args.gen_data:
        x, y = np.array(test_texts[:args.bm_samples]), np.array(test_labels[:args.bm_samples])

        total_acc, total_class = wrapmodel.total_acc_class(x, y)
        print('total_acc', total_acc)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        asa = ATGSL_SA( max_len = config.word_max_len[args.dataset], do_train= True, infer = args.infer)
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
        asa = ATGSL_SA(max_iters=1, max_len = config.word_max_len[args.dataset], infer=args.infer)
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
        bm_model = BM_model(args.bm_save_path,device)
        bsa = ATGSL_BM(max_len = config.word_max_len[args.dataset],model = bm_model, infer=True)
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
        bm_model = BM_model(args.bm_save_path, device)
        fusion_model = ATGSL_FUSION(max_len = config.word_max_len[args.dataset], model = bm_model, infer=args.infer)
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



    # if args.do_train:
    #     BM_model = train_BM(res, args)
    # else:
    #     bm_model = BM_model(args.bm_save_path)
    #


if __name__ == "__main__":
    main()

