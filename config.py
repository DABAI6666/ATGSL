class Config(object):
    num_classes = {'imdb': 2, 'yahoo': 10, 'yahoo_csv': 10, 'agnews': 4, 'fake': 2,'mr': 2,'mnli':3,'snli':3,'meituan':2,'weibo':2}
    word_max_len = {'imdb': 100, 'yahoo': 1000, 'yahoo_csv': 512, 'agnews': 150, 'fake':50,'mr': 400,'mnli':100,'snli':100,'meituan':100,'weibo':100}
    num_words = {'imdb': 5000, 'yahoo': 20000, 'yahoo_csv': 20000, 'agnews': 100000,'fake':5000,'mr':5000,'mnli':5000,'snli':5000,'meituan':5000,'weibo':5000}
    # num_words = {'imdb': 5000, 'yahoo': 20000, 'yahoo_csv': 20000, 'agnews': 5000,'fake':5000}

    wordCNN_batch_size = {'imdb': 32, 'yahoo': 32, 'yahoo_csv': 128, 'agnews': 32,'fake':32,'mr':32,'meituan':32,'weibo':32}
    wordCNN_epochs = {'imdb': 6, 'yahoo': 6, 'yahoo_csv': 5, 'agnews': 2,'fake': 4,'mr':10,'meituan':10,'weibo':10}

    bert_batch_size = {'imdb': 32, 'yahoo': 32, 'yahoo_csv': 64, 'agnews': 64,'fake':64,'mr':32,'meituan':32,'weibo':32}
    bert_epochs = {'imdb': 6, 'yahoo': 16, 'yahoo_csv': 2, 'agnews': 20,'fake':2,'mr':6,'meituan':6,'weibo':32}

    roberta_batch_size = {'imdb': 32, 'yahoo': 32, 'yahoo_csv': 64, 'agnews': 64,'fake':64,'mr':32,'meituan':32,'weibo':32}
    roberta_epochs = {'imdb': 6, 'yahoo': 16, 'yahoo_csv': 2, 'agnews': 20,'fake':2,'mr':6,'meituan':6,'weibo':32}

    LSTM_batch_size = {'imdb': 32, 'yahoo': 64, 'yahoo_csv': 128, 'agnews': 64,'mr':32,'meituan':32,'weibo':32}
    LSTM_epochs = {'imdb': 10, 'yahoo': 10, 'yahoo_csv': 10, 'agnews': 10,'mr':10,'meituan':10,'weibo':32}

    loss = {'imdb': 'binary_crossentropy', 'yahoo': 'categorical_crossentropy', 'yahoo_csv': 'categorical_crossentropy', 'agnews': 'categorical_crossentropy','fake':'binary_crossentropy','mr': 'binary_crossentropy','mnli':'binary_crossentropy','meituan': 'binary_crossentropy','weibo': 'binary_crossentropy'}
    activation = {'imdb': 'sigmoid', 'yahoo': 'softmax', 'yahoo_csv': 'softmax', 'agnews': 'softmax','fake': 'sigmoid','mr': 'sigmoid','mnli':'sigmoid','meituan':'sigmoid','weibo':'sigmoid'}

    wordCNN_embedding_dims = {'imdb': 50, 'agnews': 50,'fake':50,'mr':50,'meituan':100,'weibo':100}
    LSTM_embedding_dims = {'imdb': 50, 'agnews': 50,'fake':50,'mr':50,'meituan':100,'weibo':100}
    bert_embedding_dims = {'imdb': 50, 'agnews': 50,'fake':50,'mr':50,'mnli':50,'snli':50,'meituan':100,'weibo':100}
    roberta_embedding_dims = {'imdb': 50, 'agnews': 50,'fake':50,'mr':50,'mnli':50,'snli':50,'meituan':100,'weibo':100}

    esim_embedding_dims = {'mnli':50, 'snli':50}
    infersent_embedding_dims = {'mnli':50, 'snli':50}


    syn_word_num = 30

    #################################
    gpu = False
    num_epoch = 300
    batch_size = 16
    batch_size_test = 16

    # Word Embedding settings
    glove_directory = "../../data/glove/"
    glove_file = "glove.6B.300d.txt"
    vector_path = ""
    vocab_path = ""
    max_vocab = 20000
    emb_dim = 100
    num_structure_index = 5

    # Versioning of methods
    include_key_structure = True
    include_val_structure = True
    word_module_version = 4  # {0: max_pooling, 1: average_pooling, 2: max_pooling_w_attention, 3: average_pooling_w_attention, 4: attention}
    post_module_version = 3  # {0: average_pooling, 1: condense_into_fix_vector, 2: first_vector, 3: attention}

    # Word Embedding training
    train_word_emb = False
    train_pos_emb = False

    # Time interval embedding
    size = 100  # Number of bins
    interval = 10  # Time lapse for each interval
    include_time_interval = True

    # Word padding settings
    max_length = 35  # Pad the content to 35 words
    max_tweets = 339  # Based on the data, 339 is the largest for twitter15 and 270 is the largest for twitter16

    # Data paths
    # extension = "json"
    # data_set = "train_pheme_test_sg"
    # data_folder = "/codes/data/train_pheme_test_sg"
    # train_file_path = "train.json"
    # test_1_file_path = "test.json"
    # self.test_2_file_path = "test_small.json"

    # JSON keys --> All the keys in the JSON need to be present
    # self.keys_order = {"post_id": "id_", "label": "label", "content": "tweets", "time_delay": "time_delay",
    #                    "structure": "structure"}

    # Logs paths
    # self.dataset_name = "full_pheme"
    # self.experiment_name = "HiT_0"
    # self.log_folder = "../logs/"
    # self.record_file = "record"
    # self.interval = 100

    # Model parameters settings
    d_model = 100
    dropout_rate = 0.3

    # <------------------------ WORD LEVEL ------------------------>
    ff_word = True
    num_emb_layers_word = 2  # Model parameters settings (To encode query, key and val)
    n_mha_layers_word = 2  # Number of Multihead Attention layers
    n_head_word = 2  # Number of MHA heads

    # <------------------------ POST LEVEL ------------------------>
    ff_post = True
    num_emb_layers = 2  # Model parameters settings (To encode query, key and val)
    n_mha_layers = 12  # Number of Multihead Attention layers
    n_head = 2  # Number of MHA heads

    # Model parameters settings (For feedforward network)
    d_feed_forward = 600

    # Learning rate
    learning_rate = 0.01
    beta_1 = 0.90
    beta_2 = 0.98
    n_warmup_steps = 6000
    vary_lr = True



config = Config()
