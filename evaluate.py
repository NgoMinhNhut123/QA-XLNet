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
    ["unique_id", "start_top_log_probs", "start_top_index",
    "end_top_log_probs", "end_top_index", "cls_logits"])

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
    "start_log_prob", "end_log_prob"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

  
def decode_record(record, name_to_eval):
  eval_ex = tf.io.parse_single_example(serialized=record, features=name_to_eval)
  for name in list(eval_ex.keys()):
    t = eval_ex[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    eval_ex[name] = t
  return eval_ex

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
def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs
def write_predictions(all_examples, all_features, all_results, n_best_size,
                      output_prediction_file,
                      output_nbest_file,
                      output_null_log_odds_file, orig_data, config):
  print("writting predicttion...")
  # print("Writing predictions to: %s" % (output_prediction_file))
  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)
  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()
  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive

    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]

      cur_null_score = result.cls_logits
      # if we could have irrelevant answers, get the min score of irrelevant
      score_null = min(score_null, cur_null_score)

      for i in range(config.start_n_top):
        for j in range(config.end_n_top):
          start_log_prob = result.start_top_log_probs[i]
          start_index = result.start_top_index[i]

          j_index = i * config.end_n_top + j

          end_log_prob = result.end_top_log_probs[j_index]
          end_index = result.end_top_index[j_index]

          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= feature.paragraph_len - 1:
            continue
          if end_index >= feature.paragraph_len - 1:
            continue

          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue

          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_log_prob=start_log_prob,
                  end_log_prob=end_log_prob))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_log_prob + x.end_log_prob),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]

      tok_start_to_orig_index = feature.tok_start_to_orig_index
      tok_end_to_orig_index = feature.tok_end_to_orig_index
      start_orig_pos = tok_start_to_orig_index[pred.start_index]
      end_orig_pos = tok_end_to_orig_index[pred.end_index]

      paragraph_text = example.paragraph_text
      final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

      if final_text in seen_predictions:
        continue

      seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_log_prob=pred.start_log_prob,
              end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="", start_log_prob=-1e6,
          end_log_prob=-1e6))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None

    score_diff = score_null
    scores_diff_json[example.qas_id] = score_diff
    # note(zhiliny): always predict best_non_null_entry
    # and the evaluation script will search for the best threshold
    all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json
  
  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

  qid_to_has_ans = utils.make_qid_to_has_ans(orig_data)
  exact_raw, f1_raw = utils.get_raw_scores(orig_data, all_predictions)
  out_eval = {}
  utils.find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw,
                                   scores_diff_json, qid_to_has_ans)

  return out_eval
def load_weights(checkpoint_file,model):
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
  model.load_weights(checkpoint_file)

def main():
  config = ConfigObject()
  modelqa = modeling.XLNetModelQA(config)
  
  name_to_eval = {
    # "unique_ids": tf.io.FixedLenFeature([], tf.int64),
    "unique_ids": tf.io.FixedLenFeature([], tf.int64),
    "input_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    "input_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.float32),
    "segment_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    "cls_index": tf.io.FixedLenFeature([], tf.int64),
    "p_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.float32),

  }


  # modelqa(DUM_DATA,training=False)
  # modelqa.assign_checkpoint_to_weights()


  # modelqa.compile(optimizer='adam', loss = {'start_positions':utils.start_end_loss, 'end_positions':utils.start_end_loss,'is_impossible': utils.imposible_loss })

  checkpoint_file = os.path.join(
    config.model_dir,
    "save_model"
  )
 
  load_weights(checkpoint_file, modelqa )
  

  with tf.io.gfile.GFile(config.eval_file) as f:
      orig_data = json.load(f)["data"]
  eval_tf_file_name = os.path.join(
    config.output_dir,
    "spiece.model.0.slen-{}.qlen-{}.eval.tf_record".format(config.max_seq_length, config.max_query_length))
  eval_example_file = os.path.join(
          config.output_dir,
          "spiece.model.0.slen-{}.qlen-{}.eval.example.pkl".format(
              config.max_seq_length, config.max_query_length))
  eval_feature_file = os.path.join(
          config.output_dir,
          "spiece.model.0.slen-{}.qlen-{}.eval.feature.pkl".format(
              config.max_seq_length, config.max_query_length))
              
  if tf.io.gfile.exists(eval_example_file) and tf.io.gfile.exists(eval_feature_file):
    print("Loading eval file example...")
    with tf.io.gfile.GFile(eval_example_file, 'rb') as fin:
      eval_examples = pickle.load(fin)
    print("Loading eval file future...")
    with tf.io.gfile.GFile(eval_feature_file, 'rb') as fin:
      eval_features = pickle.load(fin)
  else:
    raise ValueError(
                  "Please run preprocess first")

  dataset_eval = tf.data.TFRecordDataset(eval_tf_file_name)
  dataset_eval = dataset_eval.apply(
      tf.data.experimental.map_and_batch(
          lambda record: decode_record(record, name_to_eval),
          batch_size=8,
          num_parallel_batches=8,
          drop_remainder=False))
  array_eval_result = []
  for step_eval, x_batch_eval in enumerate(dataset_eval):
    outputs_eval = modelqa(x_batch_eval,training=False,mode="test")
    for idx_record in range(outputs_eval['unique_ids'].shape[0]):
      array_eval_result.append(
        RawResult(
          unique_id = int(outputs_eval['unique_ids'][idx_record]),
          start_top_log_probs = [float(x) for x in outputs_eval["start_top_log_probs"][idx_record]],
          end_top_log_probs = [float(x) for x in outputs_eval["end_top_log_probs"][idx_record]],
          start_top_index = [int(x) for x in outputs_eval["start_top_index"][idx_record]],
          end_top_index = [int(x) for x in outputs_eval["end_top_index"][idx_record]],
          cls_logits= float(outputs_eval['cls_logits'][idx_record])
        )
      )
    output_prediction_file = os.path.join(
        config.predict_dir, "predictions.json")
    output_nbest_file = os.path.join(
        config.predict_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(
        config.predict_dir, "null_odds.json")

  ret = write_predictions(eval_examples, eval_features, array_eval_result, config.n_best_size, output_prediction_file, output_nbest_file, output_null_log_odds_file, orig_data,config)
  print("-----------------------------------------------")
  print("-----------------------------------------------")
  print("-----------------------------------------------")
  for key, val in ret.items():
    print(key," ----- ", val)
if __name__ == "__main__":
  main()

