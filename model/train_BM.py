
import sys,os
sys.path.append('/root/ATGSL')
sys.path.append('/home/lgy/ATGSL')

from transformers import AutoTokenizer, BertForMaskedLM
from model.convert_examples_to_features import convert_examples_to_features
import torch,argparse
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from collections import namedtuple
from tqdm import tqdm, trange
import logging, os
from config import config
from model.Attack_Classifier import BM_model


os.environ['KMP_DUPLICATE_LIB_OK']='True'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

S_info = namedtuple('S_info', 'ori_text, adv_text, mask_doc')


def train_BM(train_data, args, device):

    train_data = [S_info(_[0],_[1],_[2]) for _ in train_data]
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = BertForMaskedLM.from_pretrained(args.bert_model).to(device)
    for param in model.bert.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if name.startswith('bert.embeddings'):
            param.requires_grad = False

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # # ]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion)
    optimizer = BertAdam(model.parameters(),
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion)



    ori_lst, mask_lst, label_lst = [_.ori_text for _ in train_data], [_.mask_doc for _ in train_data], [_.adv_text for _ in train_data]



    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(ori_lst))
    logger.info("  Batch size = %d", args.train_batch_size)


    train_data = convert_examples_to_features(ori_lst, mask_lst, label_lst, bert_tokenizer, args.train_batch_size, args.infer)


    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):

        for step, batch in enumerate(tqdm(train_data, desc="Iteration")):
            input_ids,  token_type_ids, attention_mask, labels_ids = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            outputs = model(input_ids = input_ids,  token_type_ids = token_type_ids , attention_mask = attention_mask, labels = labels_ids)
            print(outputs.loss.item(),'loss')
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    bert_tokenizer.save_pretrained(args.bm_save_path)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.bm_save_path, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.bm_save_path, 'config.json')
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    # Load a trained model and config that you have fine-tuned
    # config = BertConfig(output_config_file)
    # model = BertForMaskedLM(, num_labels=num_labels)
    # model.load_state_dict(torch.load(output_model_file))
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Craft adversarial examples for a text classifier.')

    parser.add_argument('-d', '--dataset',
                        help='Dataset',
                        choices=['imdb', 'fake', 'mr', 'mnli', 'snli'],
                        default='mnli')

    parser.add_argument("--model_path", type=str, default='../victim/mnli/esim')
    parser.add_argument("--data_path", type=str, default='../data_set/mnli/')
    parser.add_argument("--bm_save_path", type=str, default='../bm_model/mnli/esim')
    parser.add_argument("--sa_data_path", type=str, default='../result/mnli/esim/sa_data.txt')

    parser.add_argument("--glove_path", type=str, default='../victim')
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--infer", type=bool, default=True)

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epoch", type=int, default=10)

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cuda",
                        type=bool,
                        default=True,
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()
    if args.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = 'cpu'
    with open(args.sa_data_path, 'r', encoding='utf-8') as f:
        train_data = eval(f.readlines()[0])
        train_data = train_data[:20]

    # train_data = [[_[0].split(' [SEP] '), _[1].split(' [SEP] '), _[2].split(' [SEP] ')] for _ in train_data]

    if args.do_train:
        model = train_BM(train_data, args, device)

    bm_model = BM_model(args.bm_save_path,device,args.infer)
    # ori_text = "This is a complex film that explores the effects of Fordist and Taylorist modes of industrial capitalist production on human relations. There are constant references to assembly line production, where workers are treated as cogs in a machine, overseen by managers wielding clipboards, controlling how much hair the workers leave exposed, and firing workers (Stanley) who meet all criteria (as his supervisor says, are always on time, are hard workers, do good work) but who may in some unspecified future make a mistake. <br /><br />This system destroys families - Stanley has to send his father to a nursing home (where he quickly dies) after Stanley loses his job. Iris' daughter is a single teen mother who drops out of high school to take a job in the plant. References are made to the fact that now, with declining wages, both partners need to work, the implication being that there's nobody left at home to care for the kids. Iris' husband is dead from an illness, and with the multiple references in the film about the costs of medical care, the viewer must wonder if he might have lived with better and more costly care. Iris' brother in law gets abusive after yet another unsuccessful day at the unemployment office when his wife yells at him for buying a beer with her savings instead of leaving it for her face lift and/or teeth job (even the working class with no stake in conventional bourgeois notions of perfection and beauty buy into them). The one reference to race in the film is through a black factory line worker whose husband is in jail (presumably, he's also black, and black men suffer disproportionally high incarceration rates). She remarks that he, like her,  - her family is composed of a prisoner and a wage slave.<br /><br />Stanley, however, still believes in human relations and is therefore for most of the film outside of the system of Fordist capitalism. He cares for his father in spite of the fact that it was his father's traveling salesman job that resulted in his illiteracy - he has not yet reduced human relations to a purely instrumental contract, as Iris' brother in law does (suggesting that he). He does not, as Iris says, conform to the work-eat-sleep routine of everyone else; rather, he uses technology and the techniques of industrial production in an artisanal and creative way, in a sort of Bauhaus ideal. This was the dream of early modernists and 1920's socialists (such as the Bauhaus) - to use technology to provide for all basic needs, allowing for more free time for creative human work and fuller human relations. He is also outside of traditional gender relations. He cooks, he cleans, he cares for his family, and he knows how to iron. Iris, on the other hand, lives in a traditionally male role - she's a factory worker, the mains source of income for her (extended) family, and she brings Stanley into the public realm, traditionally off-limits to women. By teaching him to read and write, she gives him access to the world of knowledge, also traditionally gendered male.<br /><br />Literacy here is used as a metaphor for the (traditionally masculine) public realm and the systems of circulation (monetary, vehicular, cultural) that enable participation in the public realm. Without this access, Stanley is feminized - the jobs open to him are cooking and cleaning. He is excluded from all regular circulations, unable to participate in the monetary (can't open a bank account), in the vehicular (can't get a driver's license, can't ride the bus), and in the social (he asks if he exists if he can't write his name).<br /><br />After learning to read, he grabs books on auto repair, farming, and spirituality (the Bible). The Word of God is therefore relativized, placed on the same value plane as how-to books. In fact, organized religion in general is only very occasionally present - the Bible also appears on a dresser as the camera pans to find Stanley and Iris having sex. It is, however, acknowledged as a moral force - Iris, clearly a character devoted to living a  life, mentions at the beginning of the film that her rosary was among the objects lost in a purse snatching.<br /><br />Once able to read, he enters the system and lands a managerial position with a health care plan, a car, and a house, taking his place at the head of the family, the breadwinner. Presumably, he's an industrial designer, dreaming up products that will require others enduring the drudgery of the assembly line to produce. This ending, probably the only bit of conventional Hollywood in the film, is so incongruous with all that has come before that I at least wonder if it wasn't forced in by some Studio exec suddenly worried about the lack of a feel-good ending and its potential effect on the bottom line.<br /><br />Now that, according to the pundits, we've comfortably moved on to post-industrial capitalism, the film also has a slightly nostalgic feel, as though we needed the historical distance to really analyze what happened during that period. <br /><br />Nevertheless, it's highly recommended - at least if you want to exercise your brain. Disregard the ending, and it's close to a perfect 10."
    # text = "This is a complex film that [MASK] the effects of Fordist and Taylorist modes of industrial capitalist production on human relations. There are constant references to assembly line production, where workers are treated as cogs in a machine, overseen by managers wielding clipboards, controlling how much hair the workers leave exposed, and firing workers (Stanley) who meet all criteria (as his supervisor says, are always on time, are hard workers, do good work) but who may in some unspecified future make a mistake. <br /><br />This system destroys families - Stanley has to send his father to a nursing home (where he quickly dies) after Stanley loses his job. Iris' daughter is a single teen mother who drops out of high school to take a job in the plant. References are made to the fact that now, with declining wages, both partners need to work, the implication being that there's nobody left at home to care for the kids. Iris' husband is dead from an illness, and with the multiple references in the film about the costs of medical care, the viewer must wonder if he might have lived with better and more costly care. Iris' brother in law gets abusive after yet another unsuccessful day at the unemployment office when his wife yells at him for buying a beer with her savings instead of leaving it for her face lift and/or teeth job (even the working class with no stake in conventional bourgeois notions of perfection and beauty buy into them). The one reference to race in the film is through a black factory line worker whose husband is in jail (presumably, he's also black, and black men suffer disproportionally high incarceration rates). She remarks that he, like her,  - her family is composed of a prisoner and a wage slave.<br /><br />Stanley, however, still believes in human relations and is therefore for most of the film outside of the system of Fordist capitalism. He cares for his father in spite of the fact that it was his father's traveling salesman job that resulted in his illiteracy - he has not yet reduced human relations to a purely instrumental contract, as Iris' brother in law does (suggesting that he). He does not, as Iris says, conform to the work-eat-sleep routine of everyone else; rather, he uses technology and the techniques of industrial production in an artisanal and creative way, in a sort of Bauhaus ideal. This was the dream of early modernists and 1920's socialists (such as the Bauhaus) - to use technology to provide for all basic needs, allowing for more free time for creative human work and fuller human relations. He is also outside of traditional gender relations. He cooks, he cleans, he cares for his family, and he knows how to iron. Iris, on the other hand, lives in a traditionally male role - she's a factory worker, the mains source of income for her (extended) family, and she brings Stanley into the public realm, traditionally off-limits to women. By teaching him to read and write, she gives him access to the world of knowledge, also traditionally gendered male.<br /><br />Literacy here is used as a metaphor for the (traditionally masculine) public realm and the systems of circulation (monetary, vehicular, cultural) that enable participation in the public realm. Without this access, Stanley is feminized - the jobs open to him are cooking and cleaning. He is excluded from all regular circulations, unable to participate in the monetary (can't open a bank account), in the vehicular (can't get a driver's license, can't ride the bus), and in the social (he asks if he exists if he can't write his name).<br /><br />After learning to read, he grabs books on auto repair, farming, and spirituality (the Bible). The Word of God is therefore relativized, placed on the same value plane as how-to books. In fact, organized religion in general is only very occasionally present - the Bible also appears on a dresser as the camera pans to find Stanley and Iris having sex. It is, however, acknowledged as a moral force - Iris, clearly a character devoted to living a  life, mentions at the beginning of the film that her rosary was among the objects lost in a purse snatching.<br /><br />Once able to read, he enters the system and lands a managerial position with a health care plan, a car, and a house, taking his place at the head of the family, the breadwinner. Presumably, he's an industrial designer, dreaming up products that will require others enduring the drudgery of the assembly line to produce. This ending, probably the only bit of conventional Hollywood in the film, is so incongruous with all that has come before that I at least wonder if it wasn't forced in by some Studio exec suddenly worried about the lack of a feel-good ending and its potential effect on the bottom line.<br /><br />Now that, according to the pundits, we've comfortably moved on to post-industrial capitalism, the film also has a slightly nostalgic feel, as though we needed the historical distance to really analyze what happened during that period. <br /><br />Nevertheless, it's highly recommended - at least if you want to exercise your brain. Disregard the ending, and it's close to a perfect 10."
    # ori_text = "you dangers remain shocked to discover that seinfeld 's real life is depressing"
    ori_text = "roman polanski ' s autobiographical gesture at redemption is better than ' shindler ' s list ' it is more than merely a holocaust movie"
    text =  "roman polanski ' s autobiographical gesture at [MASK] is better than ' shindler ' s list ' it is more than merely a holocaust [MASK]"
    # text = "you dangers [MASK] shocked to discover that seinfeld 's real life is depressing"
    print(bm_model.predict(ori_text, text))
# ['This site includes a list of all award winners and a searchable database of Government Executive articles. [SEP] the government executive articles housed on the website are not able to be searched .',
#  'This site includes a list of all award winners and a searchable database of Government Executive articles. [SEP] the politics executive articles housed on the website are not adept to be searched .',
#  'This site includes a list of all award winners and a searchable database of Government Executive articles. [SEP] the government executive articles housed on the website are not able to be searched .']

# nohup python -u train_BM.py -d mr --bm_save_path ../bm_model/mr  --sa_data_path ../result/mr/sa_data.txt --data_path ../data_set/mr/ --model_path ../victim/mr/cnn/cnn.dat > mr.log 2>&1 &

