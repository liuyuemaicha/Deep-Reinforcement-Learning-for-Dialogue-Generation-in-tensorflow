__author__ = 'liuyuemaicha'


class GSTConfig(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 256
    emb_dim = 1024
    num_layers = 2
    vocab_size = 25000
    train_dir = "./gst_data/"
    name_model = "st_model"
    tensorboard_dir = "./tensorboard/gst_log/"
    name_loss = "gst_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets =        [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]


class GCCConfig(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 128
    emb_dim = 1024
    num_layers = 2
    vocab_size = 25000
    train_dir = "./gcc_data/"
    name_model = "cc_model"
    tensorboard_dir = "./tensorboard/gcc_log/"
    name_loss = "gcc_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets =        [(10, 10), (20, 15), (40, 25), (80, 50)]
    buckets_concat = [(10, 10), (20, 15), (40, 25), (80, 50), (100,50)]


class GBKConfig(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 256
    emb_dim = 1024
    num_layers = 2
    vocab_size = 25000
    train_dir = "./gbk_data/"
    name_model = "bk_model"
    tensorboard_dir = "./tensorboard/gbk_log/"
    name_loss = "gbk_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets =        [(10, 5), (15, 10), (25, 20), (50, 40)]
    buckets_concat = [(10, 5), (15, 10), (25, 20), (50, 40), (100, 50)]


class GRLConfig(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 256
    emb_dim = 1024
    num_layers = 2
    vocab_size = 25000
    train_dir = "./grl_data/"
    name_model = "rl_model"
    tensorboard_dir = "./tensorboard/grl_log/"
    name_loss = "grl_loss"
    pre_name_loss = "pre_rl_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets =        [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]

class Pre_GRLConfig(object):
    beam_size = 4
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 10
    emb_dim = 512
    num_layers = 2
    vocab_size = 1000
    train_dir = "./pre_grl_data/"
    name_model = "rl_model"
    tensorboard_dir = "./tensorboard/grl_log/"
    name_loss = "grl_loss"
    pre_name_loss = "pre_rl_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets =        [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]