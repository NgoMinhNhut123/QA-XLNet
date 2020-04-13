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

import numpy as np

if six.PY2:
  import cPickle as pickle
else:
  import pickle

import tensorflow as tf
import sentencepiece as spm
import modeling
from config import ConfigObject
from  utils import preprocess_text, encode_ids, encode_pieces, printable_text
SPIECE_UNDERLINE = u'▁'

SEG_ID_P   = 0
SEG_ID_Q   = 1
SEG_ID_CLS = 2
SEG_ID_PAD = 3
special_symbols = {
    "<unk>"  : 0,
    "<s>"    : 1,
    "</s>"   : 2,
    "<cls>"  : 3,
    "<sep>"  : 4,
    "<pad>"  : 5,
    "<mask>" : 6,
    "<eod>"  : 7,
    "<eop>"  : 8,
}
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]

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


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.compat.v1.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]
  if is_training:
    examples_train = []
    examples_test = []
    check = 0
  else:
    examples_eval = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        orig_answer_text = None
        is_impossible = False

        if is_training:
          is_impossible = qa["is_impossible"]
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            start_position = answer["answer_start"]
          else:
            start_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            paragraph_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            is_impossible=is_impossible)
        if is_training:
          if check%12 == 0:
            examples_test.append(example)
          else:
            examples_train.append(example)
          check = check+1
        else:
          examples_eval.append(example)
  if is_training:
    return examples_test , examples_train
  else:
    return examples_eval
class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["p_mask"] = create_float_feature(feature.p_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    features["cls_index"] = create_int_feature([feature.cls_index])

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_float_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def _convert_index(index, pos, M=None, is_start=True):
  if index[pos] is not None:
    return index[pos]
  N = len(index)
  rear = pos
  while rear < N - 1 and index[rear] is None:
    rear += 1
  front = pos
  while front > 0 and index[front] is None:
    front -= 1
  assert index[front] is not None or index[rear] is not None
  if index[front] is None:
    if index[rear] >= 1:
      if is_start:
        return 0
      else:
        return index[rear] - 1
    return index[rear]
  if index[rear] is None:
    if M is not None and index[front] < M - 1:
      if is_start:
        return index[front] + 1
      else:
        return M - 1
    return index[front]
  if is_start:
    if index[rear] > index[front] + 1:
      return index[front] + 1
    else:
      return index[rear]
  else:
    if index[rear] > index[front] + 1:
      return index[rear] - 1
    else:
      return index[front]

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index





def convert_examples_to_features(config,examples, sp_model, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  print('reading and save recored ....')
  cnt_pos, cnt_neg = 0, 0
  unique_id = 1000000000
  max_N, max_M = 1024, 1024
  f = np.zeros((max_N, max_M), dtype=np.float32)
  for (example_index, example) in enumerate(examples):

    if example_index % 100 == 0:
      print('Converting {}/{} pos {} neg {}'.format(
          example_index, len(examples), cnt_pos, cnt_neg))

    query_tokens = encode_ids(
        sp_model,
        preprocess_text(example.question_text, lower=config.uncased))

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    paragraph_text = example.paragraph_text
    para_tokens = encode_pieces(
        sp_model,
        preprocess_text(example.paragraph_text, lower=config.uncased))

    chartok_to_tok_index = []
    tok_start_to_chartok_index = []
    tok_end_to_chartok_index = []
    char_cnt = 0
    for i, token in enumerate(para_tokens):
      chartok_to_tok_index.extend([i] * len(token))
      tok_start_to_chartok_index.append(char_cnt)
      char_cnt += len(token)
      tok_end_to_chartok_index.append(char_cnt - 1)

    tok_cat_text = ''.join(para_tokens).replace(SPIECE_UNDERLINE, ' ')
    N, M = len(paragraph_text), len(tok_cat_text)

    if N > max_N or M > max_M:
      max_N = max(N, max_N)
      max_M = max(M, max_M)
      f = np.zeros((max_N, max_M), dtype=np.float32)
      gc.collect()

    g = {}

    def _lcs_match(max_dist):
      f.fill(0)
      g.clear()

      ### longest common sub sequence
      # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
      for i in range(N):

        # note(zhiliny):
        # unlike standard LCS, this is specifically optimized for the setting
        # because the mismatch between sentence pieces and original text will
        # be small
        for j in range(i - max_dist, i + max_dist):
          if j >= M or j < 0: continue

          if i > 0:
            g[(i, j)] = 0
            f[i, j] = f[i - 1, j]

          if j > 0 and f[i, j - 1] > f[i, j]:
            g[(i, j)] = 1
            f[i, j] = f[i, j - 1]

          f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
          if (preprocess_text(paragraph_text[i], lower=config.uncased,
              remove_space=False)
              == tok_cat_text[j]
              and f_prev + 1 > f[i, j]):
            g[(i, j)] = 2
            f[i, j] = f_prev + 1

    max_dist = abs(N - M) + 5
    for _ in range(2):
      _lcs_match(max_dist)
      if f[N - 1, M - 1] > 0.8 * N: break
      max_dist *= 2

    orig_to_chartok_index = [None] * N
    chartok_to_orig_index = [None] * M
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
      if (i, j) not in g: break
      if g[(i, j)] == 2:
        orig_to_chartok_index[i] = j
        chartok_to_orig_index[j] = i
        i, j = i - 1, j - 1
      elif g[(i, j)] == 1:
        j = j - 1
      else:
        i = i - 1

    if all(v is None for v in orig_to_chartok_index) or f[N - 1, M - 1] < 0.8 * N:
      
      print('MISMATCH DETECTED!')
      continue

    tok_start_to_orig_index = []
    tok_end_to_orig_index = []
    for i in range(len(para_tokens)):
      start_chartok_pos = tok_start_to_chartok_index[i]
      end_chartok_pos = tok_end_to_chartok_index[i]
      start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                      N, is_start=True)
      end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
                                    N, is_start=False)

      tok_start_to_orig_index.append(start_orig_pos)
      tok_end_to_orig_index.append(end_orig_pos)

    if not is_training:
      tok_start_position = tok_end_position = None

    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1

    if is_training and not example.is_impossible:
      start_position = example.start_position
      end_position = start_position + len(example.orig_answer_text) - 1

      start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                         is_start=True)
      tok_start_position = chartok_to_tok_index[start_chartok_pos]

      end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                       is_start=False)
      tok_end_position = chartok_to_tok_index[end_chartok_pos]
      assert tok_start_position <= tok_end_position

    def _piece_to_id(x):
      if six.PY2 and isinstance(x, unicode):
        x = x.encode('utf-8')
      return sp_model.PieceToId(x)

    all_doc_tokens = list(map(_piece_to_id, para_tokens))

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_is_max_context = {}
      segment_ids = []
      p_mask = []

      cur_tok_start_to_orig_index = []
      cur_tok_end_to_orig_index = []

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i

        cur_tok_start_to_orig_index.append(
            tok_start_to_orig_index[split_token_index])
        cur_tok_end_to_orig_index.append(
            tok_end_to_orig_index[split_token_index])

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(SEG_ID_P)
        p_mask.append(0)

      paragraph_len = len(tokens)

      tokens.append(SEP_ID)
      segment_ids.append(SEG_ID_P)
      p_mask.append(1)

      # note(zhiliny): we put P before Q
      # because during pretraining, B is always shorter than A
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(SEG_ID_Q)
        p_mask.append(1)
      tokens.append(SEP_ID)
      segment_ids.append(SEG_ID_Q)
      p_mask.append(1)

      cls_index = len(segment_ids)
      tokens.append(CLS_ID)
      segment_ids.append(SEG_ID_CLS)
      p_mask.append(0)

      input_ids = tokens

      # The mask has 0 for real tokens and 1 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [0] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(1)
        segment_ids.append(SEG_ID_PAD)
        p_mask.append(1)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(p_mask) == max_seq_length

      span_is_impossible = example.is_impossible
      start_position = None
      end_position = None
      if is_training and not span_is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          # continue
          start_position = 0
          end_position = 0
          span_is_impossible = True
        else:
          # note(zhiliny): we put P before Q, so doc_offset should be zero.
          # doc_offset = len(query_tokens) + 2
          doc_offset = 0
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and span_is_impossible:
        start_position = cls_index
        end_position = cls_index

      if example_index < 20:
        print("*** Example ***")
        print("unique_id: %s" % (unique_id))
        print("example_index: %s" % (example_index))
        print("doc_span_index: %s" % (doc_span_index))
        print("tok_start_to_orig_index: %s" % " ".join(
            [str(x) for x in cur_tok_start_to_orig_index]))
        print("tok_end_to_orig_index: %s" % " ".join(
            [str(x) for x in cur_tok_end_to_orig_index]))
        print("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        if is_training and span_is_impossible:
          print("impossible example span")

        if is_training and not span_is_impossible:
          pieces = [sp_model.IdToPiece(token) for token in
                    tokens[start_position: (end_position + 1)]]
          answer_text = sp_model.DecodePieces(pieces)
          print("start_position: %d" % (start_position))
          print("end_position: %d" % (end_position))
          print(
              "answer: %s" % (printable_text(answer_text)))

          # note(zhiliny): With multi processing,
          # the example_index is actually the index within the current process
          # therefore we use example_index=None to avoid being used in the future.
          # The current code does not use example_index of training data.
      
      feat_example_index = example_index

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=feat_example_index,
          doc_span_index=doc_span_index,
          tok_start_to_orig_index=cur_tok_start_to_orig_index,
          tok_end_to_orig_index=cur_tok_end_to_orig_index,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          p_mask=p_mask,
          segment_ids=segment_ids,
          paragraph_len=paragraph_len,
          cls_index=cls_index,
          start_position=start_position,
          end_position=end_position,
          is_impossible=span_is_impossible)

      # Run callback
      
      output_fn(feature)

      unique_id += 1
      if span_is_impossible:
        cnt_neg += 1
      else:
        cnt_pos += 1





def main():
  config = ConfigObject()
  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(config.spiece_model_file)


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
  train_tf_file_name = os.path.join(
    config.output_dir,
    "spiece.model.0.slen-{}.qlen-{}.train.tf_record".format(config.max_seq_length, config.max_query_length))
  test_tf_file_name  = os.path.join(
    config.output_dir,
    "spiece.model.0.slen-{}.qlen-{}.test.tf_record".format(config.max_seq_length, config.max_query_length))
  # eval_tf_file_name = "spiece.model.0.slen-{}.qlen-{}.eval.tf_record".format(config.max_seq_length, config.max_query_length)
  test_example_file = os.path.join(
          config.output_dir,
          "spiece.model.0.slen-{}.qlen-{}.test.example.pkl".format(
              config.max_seq_length, config.max_query_length))
  test_feature_file = os.path.join(
          config.output_dir,
          "spiece.model.0.slen-{}.qlen-{}.test.feature.pkl".format(
              config.max_seq_length, config.max_query_length))


  examples_test, examples_train = read_squad_examples(config.train_file,True)
  examples_eval = read_squad_examples(config.eval_file,False)
  random.shuffle(examples_train)

  with tf.io.gfile.GFile(test_example_file, 'wb') as fout:
    pickle.dump(examples_test, fout)
  with tf.io.gfile.GFile(eval_example_file, 'wb') as fout:
    pickle.dump(examples_eval, fout)


  train_writer = FeatureWriter(
      filename=train_tf_file_name,
      is_training=True)
  convert_examples_to_features(
        config=config,
        examples=examples_train,
        sp_model=sp_model,
        max_seq_length=config.max_seq_length,
        doc_stride=config.doc_stride,
        max_query_length=config.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)
  train_writer.close()


  dev_writer = FeatureWriter(
      filename=test_tf_file_name,
      is_training=True)
  test_features = []
  def append_test_feature(feature):
          test_features.append(feature)
          dev_writer.process_feature(feature)
  convert_examples_to_features(
        config=config,
        examples=examples_test,
        sp_model=sp_model,
        max_seq_length=config.max_seq_length,
        doc_stride=config.doc_stride,
        max_query_length=config.max_query_length,
        is_training=True,
        output_fn=append_test_feature)
  dev_writer.close()

  eval_writer = FeatureWriter(
      filename=eval_tf_file_name,
      is_training=False)
  eval_features = []
  def append_eval_feature(feature):
          eval_features.append(feature)
          eval_writer.process_feature(feature)
  convert_examples_to_features(
        config=config,
        examples=examples_eval,
        sp_model=sp_model,
        max_seq_length=config.max_seq_length,
        doc_stride=config.doc_stride,
        max_query_length=config.max_query_length,
        is_training=False,
        output_fn=append_eval_feature)
  eval_writer.close()


  with tf.io.gfile.GFile(test_feature_file, 'wb') as fout:
          pickle.dump(test_features, fout)
  with tf.io.gfile.GFile(eval_feature_file, 'wb') as fout:
          pickle.dump(eval_features, fout)
  # create_dev_and_train(config,train_dev_examples, train_tf_file_name,  test_tf_file_name)
  # print("Write to {}".format(train_tf_file_name))

if __name__ == "__main__":
  main()