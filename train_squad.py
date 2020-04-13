# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import collections
import os
import time
import math
import json
import six
import random
import gc
import re
import numpy as np

if six.PY2:
  import cPickle as pickle
else:
  import pickle

import tensorflow as tf
import sentencepiece as spm
import modeling
import utils
from config import ConfigObject
import time
import datetime
# from preprocess import InputFeatures
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length.")
flags.DEFINE_string("model_config_path", default=None,
      help="Model config path.")
flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("predict_dir", default="",
                    help="Dir for predictions.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("train_file", default="",
                    help="Path of train file.")
flags.DEFINE_string("predict_file", default="",
                    help="Path of prediction file.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                    "Could be a pretrained model or a finetuned model.")
flags.DEFINE_integer("max_seq_length",
                    default=512, help="Max sequence length")
flags.DEFINE_integer("max_query_length",
                    default=64, help="Max query length")
flags.DEFINE_integer("doc_stride",
                    default=128, help="Doc stride")
flags.DEFINE_integer("max_answer_length",
                    default=64, help="Max answer length")
flags.DEFINE_bool("uncased", default=False, help="Use uncased data.")

FLAGS = flags.FLAGS


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tok_start_to_orig_index,
               tok_end_to_orig_index,
               token_is_max_context,
               input_ids,
               input_mask,
               p_mask,
               segment_ids,
               paragraph_len,
               cls_index,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tok_start_to_orig_index = tok_start_to_orig_index
    self.tok_end_to_orig_index = tok_end_to_orig_index
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.p_mask = p_mask
    self.segment_ids = segment_ids
    self.paragraph_len = paragraph_len
    self.cls_index = cls_index
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               paragraph_text,
               orig_answer_text=None,
               start_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.paragraph_text = paragraph_text
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (printable_text(self.qas_id))
    s += ", question_text: %s" % (
        printable_text(self.question_text))
    s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s

RawResult = collections.namedtuple("RawResult",
    ["unique_id", "start_index", "start_log_prob",
    "end_index", "end_log_prob", "cls_logits"])
_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
    "start_log_prob", "end_log_prob"])

def compute_scores(array_test_result, test_examples , test_features):
  print("   compting_scores...")
  example_index_to_features = collections.defaultdict(list)
  unique_id_to_result = {}
  for result in array_test_result:
    unique_id_to_result[result.unique_id] = result
  
  for feature in test_features:
    if feature.unique_id in unique_id_to_result:
      example_index_to_features[feature.example_index].append(feature)
  text_predictions = collections.OrderedDict()
  scores_diff = collections.OrderedDict()
  for (example_index, example) in enumerate(test_examples):
    if example_index in example_index_to_features:
      features = example_index_to_features[example_index]
      score_null = 1000000
      prelim_predictions = []
      text_predict = ""
      for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        cur_null_score = result.cls_logits
        score_null = min(score_null, cur_null_score)
        if not feature.token_is_max_context.get(result.start_index, False):
            continue
        if result.end_index <  result.start_index:
          continue
        if result.start_index >=  len(feature.tok_start_to_orig_index) or result.end_index >=  len(feature.tok_end_to_orig_index):
          continue
        if result.start_index >= feature.paragraph_len - 1:
          continue
        if result.end_index >= feature.paragraph_len - 1:
          continue
        prelim_predictions.append(
          _PrelimPrediction(
            feature_index = feature_index,
            start_index = result.start_index,
            end_index = result.end_index,
            start_log_prob = result.start_log_prob,
            end_log_prob =result.end_log_prob
          )
        )
      prelim_predictions = sorted(
        prelim_predictions,
        key = lambda x: (x.start_log_prob + x.end_log_prob),
        reverse = True
      )
      if prelim_predictions:
        predict_max = prelim_predictions[0]
        feature = features[predict_max.feature_index]
        start_orig_pos = feature.tok_start_to_orig_index[predict_max.start_index]
        end_orig_pos = feature.tok_end_to_orig_index[predict_max.end_index]
        text_predict = example.paragraph_text[start_orig_pos:end_orig_pos + 1].strip()
      scores_diff[example.qas_id]  = score_null
      text_predictions[example.qas_id]= text_predict 

  
  out_test={}
  utils.get_score_dev(out_test,test_examples, text_predictions, scores_diff)
  return out_test
  

def assign_premodel_checkpoint_to_weights(model, file_checkpoint) :
  input_ids =tf.ones([4,32],dtype=tf.int64)
  input_mask = tf.ones([4,32],dtype=tf.float32)
  segment_ids =tf.ones([4,32],dtype=tf.int32)
  p_mask= tf.ones([4,32],dtype=tf.float32)
  cls_index=tf.ones([4],dtype=tf.int32)
  start_positions=tf.ones([4],dtype=tf.int32)
  DUM_DATA = {
    "input_ids": input_ids, 
    "input_mask":input_mask, 
    "segment_ids":segment_ids, 
    "cls_index":cls_index,
    "p_mask": p_mask,
    "start_positions": start_positions
  }
  model(inputs=DUM_DATA, training=True, mode="train")
  weights = model.get_weights()
  ckpt_reader = tf.train.load_checkpoint(file_checkpoint)
  var_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  names = [weight.name for layer in model.layers for weight in layer.weights]
  print("Finetunning squad have {} weights".format(len(weights)))
  i=0
  array_numpy=[]
  for name, weight in zip(names, weights):
    if any(name[:-2] in key for key in var_to_shape_map):
      i=i+1
      array_numpy.append(ckpt_reader.get_tensor(name[:-2]))
      print(" Get weight {} from TFv1 Pre-model to TFv2 model with shape {} ".format(name[:-2], ckpt_reader.get_tensor(name[:-2]).shape))
      # print(name[:-2], weight)
    else:
      array_numpy.append(weight)
  print("Get {} weights from checkpoint premodel".format(i))
  
  print("--------------------------------------------------------------------")
  print("--------------------------------------------------------------------")
  print("--------------------------------------------------------------------")
  print("--------------------------------------------------------------------")
  print("--------------------------------------------------------------------")
  model.set_weights(array_numpy)
  return
  
def decode_record(record, name_to_train, name_to_output,config):
  train_ex = tf.io.parse_single_example(serialized=record, features=name_to_train)
  output_ex = tf.io.parse_single_example(serialized=record, features=name_to_output)
  for name in list(train_ex.keys()):
    t = train_ex[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    train_ex[name] = t
  for name in list(output_ex.keys()):
    t = output_ex[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    output_ex[name] = t
  output_ex["start_positions"]  = tf.one_hot(output_ex["start_positions"], depth=config.max_seq_length, dtype=tf.float32)
  output_ex["end_positions"]  = tf.one_hot( output_ex["end_positions"], depth=config.max_seq_length, dtype=tf.float32)
  return train_ex , output_ex

# {
#     "d_head": 64, 
#     "d_inner": 3072, 
#     "d_model": 768, 
#     "ff_activation": "gelu", 
#     "n_head": 12, 
#     "n_layer": 12, 
#     "n_token": 32000, 
#     "untie_r": true
# }
def train_model(config, modelqa, dataset_train,dataset_dev, loss_summary_writer,score_test_summary_writer):
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate= config.lr,
      decay_steps = config.train_global_step - config.warmup_steps,
      end_learning_rate= config.lr * config.min_lr_ratio
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=config.adam_epsilon)
    glo_step = 0
    for epoch in range(5):
      print("start epoch {}".format(epoch))
      print("--------------------------------------------------------------------")
      print("--------------------------------------------------------------------")
      for step_train, (x_batch_train, y_batch_train) in enumerate(dataset_train):
        with tf.GradientTape() as tape:
          outputs = modelqa(x_batch_train,training=True,mode="train")
          loss_start = utils.start_end_loss(y_batch_train['start_positions'],outputs['start_log_probs'] )
          loss_end = utils.start_end_loss(y_batch_train['end_positions'],outputs['end_log_probs'])
          loss_impossible = utils.imposible_loss(y_batch_train['is_impossible'], outputs['cls_logits'])
          total_loss = loss_impossible + loss_start + loss_end
        grads = tape.gradient(total_loss, modelqa.trainable_weights)
        clipped, gnorm = tf.clip_by_global_norm(grads, config.clip)
        
        if getattr(config, "lr_layer_decay_rate", 1.0) != 1.0:
          n_layer = 0
          names = [weight.name for layer in modelqa.layers for weight in layer.weights]
          # print(len(modelqa.trainable_weights) == len(names),"vvvvvvvvvvvvvvv" )
          # print([i for i in names])
          for i in range(len(clipped)):
            m = re.search(r"model/transformer/layer_(\d+?)/", names[i])
            if not m: continue
            n_layer = max(n_layer, int(m.group(1)) + 1)
          for i in range(len(clipped)):
            for l in range(n_layer):
              if "model/transformer/layer_{}/".format(l) in names[i]:
                abs_rate = config.lr_layer_decay_rate ** (n_layer - 1 - l)
                clipped[i] *= abs_rate
                break

        
        optimizer.apply_gradients(zip(clipped, modelqa.trainable_weights))
        with loss_summary_writer.as_default():
          tf.summary.scalar('loss/train', total_loss, step=glo_step)
        if step_train == 0:
          start_time = time.time()
        if step_train%200 == 0:
          end_time =time.time()
          print("Training loss (for one batch) at step {} - epoch {} - globalStep -{} ({:.2f}s) {:.2f}s/step : {} \n loss_start: {} loss_end: {} loss_impossible: {}".format(step_train,epoch,glo_step,end_time-start_time,(end_time-start_time)/200,total_loss,loss_start,loss_end,loss_impossible))
          start_time =time.time()
        if step_train%5000 == 0:
          average_loss = 0
          len_loss = 0
          array_test_result = []
          for step_val, (x_batch_val, y_batch_val) in enumerate(dataset_dev):
            outputs_dev = modelqa(x_batch_val,training=True,mode="test")
            for idx_record in range(outputs_dev['unique_ids'].shape[0]):
              array_test_result.append(
                RawResult(
                  unique_id = int(outputs_dev['unique_ids'][idx_record]),
                  start_index = int(np.argmax(outputs_dev['start_log_probs'][idx_record])),
                  start_log_prob = float(np.amax(outputs_dev['start_log_probs'][idx_record])),
                  end_index = int(np.argmax(outputs_dev['end_log_probs'][idx_record])),
                  end_log_prob = float(np.amax(outputs_dev['end_log_probs'][idx_record])),
                  cls_logits= float(outputs_dev['cls_logits'][idx_record])
                )
              )
            loss_start_dev = utils.start_end_loss(y_batch_val['start_positions'],outputs_dev['start_log_probs'] )
            loss_end_dev = utils.start_end_loss(y_batch_val['end_positions'],outputs_dev['end_log_probs'])
            loss_impossible_dev = utils.imposible_loss(y_batch_val['is_impossible'], outputs_dev['cls_logits'])
            total_loss_dev = loss_impossible_dev + loss_start_dev + loss_end_dev
            average_loss = average_loss + total_loss_dev
            len_loss = len_loss + 1
            if step_val == 0:
              start_time_dev = time.time()
            if step_val%200 == 0:
              end_time_dev =time.time()
              print("--------------------------------------------------------------------")
              print("   Testing loss (for one batch) at step {} - epoch {} - globalStep -{} ({:.2f}s) {:.2f}s/step : {} \n    loss_start: {} loss_end: {} loss_impossible: {}".format(step_val,epoch,glo_step,end_time_dev-start_time_dev,(end_time_dev-start_time_dev)/200,total_loss_dev,loss_start_dev,loss_end_dev,loss_impossible_dev))
              start_time_dev =time.time()
            
          average_loss = average_loss / len_loss
          out_test = compute_scores(array_test_result, test_examples , test_features)
          print("   loss_average = {} , f1 = {}, em= {}".format(average_loss,out_test['best_f1'],out_test['best_exact']))
          with loss_summary_writer.as_default():
            tf.summary.scalar('loss/dev', average_loss, step=glo_step)
          with score_test_summary_writer.as_default():
            tf.summary.scalar('f1/dev', out_test['best_f1'], step=glo_step)
          with score_test_summary_writer.as_default():
            tf.summary.scalar('em/dev', out_test['best_exact'], step=glo_step)  
        glo_step = glo_step + 1
        if glo_step >= config.train_global_step:
          modelqa.save_weights(checkpoint_file)
          return
def main():
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  loss_log_dir = 'logs/gradient_tape/' + current_time + '/train'
  # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
  score_test_log_dir = 'logs/gradient_tape/' + current_time + '/score_test'
  # em_test_log_dir = 'logs/gradient_tape/' + current_time + '/em_test'
  loss_summary_writer = tf.summary.create_file_writer(loss_log_dir)
  score_test_summary_writer = tf.summary.create_file_writer(score_test_log_dir)
  # f1_test_summary_writer = tf.summary.create_file_writer(f1_test_log_dir)
  # em_test_summary_writer = tf.summary.create_file_writer(em_test_log_dir)

  config = ConfigObject()
  modelqa = modeling.XLNetModelQA(config)
  name_to_train = {
    # "unique_ids": tf.io.FixedLenFeature([], tf.int64),
    "input_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    "input_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.float32),
    "segment_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    "cls_index": tf.io.FixedLenFeature([], tf.int64),
    "p_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.float32),
    "start_positions": tf.io.FixedLenFeature([], tf.int64)

  }
  name_to_dev = {
    # "unique_ids": tf.io.FixedLenFeature([], tf.int64),
    "unique_ids": tf.io.FixedLenFeature([], tf.int64),
    "input_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    "input_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.float32),
    "segment_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    "cls_index": tf.io.FixedLenFeature([], tf.int64),
    "p_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.float32),

  }
  name_to_output = {
    "start_positions": tf.io.FixedLenFeature([], tf.int64),
    "end_positions": tf.io.FixedLenFeature([], tf.int64),
    "is_impossible": tf.io.FixedLenFeature([], tf.float32)

  }

  # modelqa(DUM_DATA,training=False)
  # modelqa.assign_checkpoint_to_weights()


  # modelqa.compile(optimizer='adam', loss = {'start_positions':utils.start_end_loss, 'end_positions':utils.start_end_loss,'is_impossible': utils.imposible_loss })

  checkpoint_file = os.path.join(
    config.model_dir,
    "save_model"
  )
  tf_file_train = os.path.join(
    config.output_dir,
    "spiece.model.0.slen-{}.qlen-{}.train.tf_record".format(config.max_seq_length, config.max_query_length)
  )
  tf_file_test = os.path.join(
    config.output_dir,
    "spiece.model.0.slen-{}.qlen-{}.test.tf_record".format(config.max_seq_length, config.max_query_length)
  )
  test_example_file = os.path.join(
          config.output_dir,
          "spiece.model.0.slen-{}.qlen-{}.test.example.pkl".format(
              config.max_seq_length, config.max_query_length))
  test_feature_file = os.path.join(
          config.output_dir,
          "spiece.model.0.slen-{}.qlen-{}.test.feature.pkl".format(
              config.max_seq_length, config.max_query_length))

  if tf.io.gfile.exists(test_example_file) and tf.io.gfile.exists(test_feature_file):
    print("Loading test file example...")
    with tf.io.gfile.GFile(test_example_file, 'rb') as fin:
      test_examples = pickle.load(fin)
    print("Loading test file future...")
    with tf.io.gfile.GFile(test_feature_file, 'rb') as fin:
      test_features = pickle.load(fin)
  else:
    raise ValueError(
                  "Please run preprocess first")

  dataset_train = tf.data.TFRecordDataset(tf_file_train)
  dataset_train = dataset_train.shuffle(2048)
  dataset_train = dataset_train.repeat()
  dataset_train = dataset_train.apply(
      tf.data.experimental.map_and_batch(
          lambda record: decode_record(record, name_to_train , name_to_output,config),
          batch_size=4,
          num_parallel_batches=8,
          drop_remainder=True))
  dataset_train = dataset_train.prefetch(1024)


  dataset_dev = tf.data.TFRecordDataset(tf_file_test)
  dataset_dev = dataset_dev.apply(
      tf.data.experimental.map_and_batch(
          lambda record: decode_record(record, name_to_dev , name_to_output,config),
          batch_size=8,
          num_parallel_batches=8,
          drop_remainder=False))


  assign_premodel_checkpoint_to_weights(modelqa,config.init_checkpoint)
  

  # for step_val, (x_batch_val, y_batch_val) in enumerate(dataset_dev):
  #   print(x_batch_val["unique_ids"])

  # print(len(modelqa.trainable_weights))
  

  train_model(config, modelqa, dataset_train,dataset_dev, loss_summary_writer,score_test_summary_writer)
  
  print("end")
if __name__ == "__main__":
  main()

