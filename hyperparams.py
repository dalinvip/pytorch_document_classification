# @Author : bamtercelboo
# @Datetime : 2018/1/14 22:45
# @File : hyperparams.py
# @Last Modify Time : 2018/1/14 22:45
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  hyperparams.py
    FUNCTION : set hyperparams for load Data、model、train
"""
import torch
import random
# random seed num
seed_num = 532
torch.manual_seed(seed_num)
random.seed(seed_num)


class Hyperparams():
    def __init__(self):
        # Datasets
        # IMDB
        self.IMDB = True
        self.train_path = "./Data/IMDB_debug/imdb_trainall_1.txt"
        self.dev_path = None
        self.test_path = "./Data/IMDB_debug/imdb_testall_1.txt"
        self.CV = False
        # self.train_path = "./Data/MR/rt-polarity.all"
        # self.train_path = "./Data/MPQA/mpqa.all"
        # self.train_path = "./Data/CR/custrev.all"
        # self.train_path = "./Data/Subj/subj.all"
        # self.dev_path = None
        # self.test_path = None

        self.shuffle = True
        self.epochs_shuffle = True
        self.nfold = 10

        # model
        self.SumPooling = True
        self.embed_dim = 100
        self.dropout = 0.5
        self.dropout_embed = 0.3
        self.clip_max_norm = 5

        # select optim algorhtim for train
        self.Adam = True
        self.learning_rate = 0.001
        self.learning_rate_decay = 1   # value is 1 means not change lr
        # L2 weight_decay
        self.weight_decay = 1e-8  # default value is zero in Adam SGD
        # self.weight_decay = 0   # default value is zero in Adam SGD
        self.epochs = 1000
        self.train_batch_size = 16
        self.dev_batch_size = None  # "None meaning not use batch for dev"
        self.test_batch_size = None  # "None meaning not use batch for test"
        self.log_interval = 1
        self.dev_interval = 100
        self.test_interval = 100
        self.save_dir = "snapshot"
        # whether to delete the model after test acc so that to save space
        self.rm_model = True

        # min freq to include during built the vocab, default is 1
        self.min_freq = 1

        # word_Embedding
        self.word_Embedding = False
        # self.word_Embedding_Path = "./Pretrain_Embedding/richfeat/sentence_classification/enwiki.emb.feat30_TREC.txt"
        self.word_Embedding_Path = "/home/lzl/mszhang/suda_file0120/sentence_classification_richfeat/enwiki.emb.source_feat_SST1.txt"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file_0113/file/context/sentence_classification/enwiki.emb.source_CR.txt"
        # self.word_Embedding_Path = "/home/lzl/mszhang/suda_file_0113/file/context/enwiki.emb.source_CR.txt"

        # GPU
        self.use_cuda = False
        self.gpu_device = -1  # -1 meaning no use cuda
        self.num_threads = 1
