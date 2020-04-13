import tensorflow as tf
class ConfigObject(object):
  def __init__(self):
    self.n_best_size = 5
    self.start_n_top = 5
    self.end_n_top = 5
    self.dropout = 0.1
    self.dropatt = 0.1
    self.attn_type  = 'bi'
    self.clamp_len = -1
    self.initializer = tf.random_normal_initializer( stddev=0.02,seed =None)
    self.ff_activation= 'relu'
    self.bi_data = False
    self.mem_len = None
    self.same_length = False
    self.reuse_len = None
    self.n_token = 32000
    self.n_layer = 12
    self.d_model = 768
    self.n_head = 12
    self.d_head = 64
    self.d_inner = 3072
    self.init_checkpoint = "xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt"
    self.batch_size = 4
    self.max_seq_length=340
    self.max_query_length=64
    self.output_dir= 'proc_data/squad'
    self.spiece_model_file = 'xlnet_cased_L-12_H-768_A-12/spiece.model'
    self.train_file = 'data/squad/train-v2.0.json'
    self.eval_file = 'data/squad/dev-v2.0.json'
    self.doc_stride = 128
    self.do_eval = False
    self.uncased =False
    self.epochs = 3
    self.lr = 3e-5
    self.adam_epsilon = 1e-6
    self.lr_layer_decay_rate =0.75
    self.train_global_step = 80000
    self.min_lr_ratio = 0.0
    self.model_dir = "experiment/squad"
    self.clip = 1.0
    self.warmup_steps=1000
    self.predict_dir = 'predict/squad'