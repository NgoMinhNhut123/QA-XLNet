# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata
import six
from functools import partial
import sys
import re
import numpy as np
import collections
import string
SPIECE_UNDERLINE = '▁'



import tensorflow as tf



def imposible_loss(y_actual,y_pred):
  regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=y_actual, logits=y_pred)
  regression_loss = tf.reduce_mean(input_tensor=regression_loss)
  return regression_loss*0.5


def start_end_loss(y_actual,y_pred):
  loss = - tf.reduce_sum(input_tensor=y_actual * y_pred, axis=-1)
  loss = tf.reduce_mean(input_tensor=loss)
  return loss*0.5


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs
  outputs = outputs.replace("``", '"').replace("''", '"')

  if six.PY2 and isinstance(outputs, str):
    outputs = outputs.decode('utf-8')

  if not keep_accents:
    outputs = unicodedata.normalize('NFKD', outputs)
    outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def encode_ids(sp_model, text, sample=False):
  pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids

def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  # return_unicode is used only for py2

  # note(zhiliny): in some systems, sentencepiece only accepts str for py2
  if six.PY2 and isinstance(text, unicode):
    text = text.encode('utf-8')

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
  new_pieces = []
  for piece in pieces:
    if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          piece[:-1].replace(SPIECE_UNDERLINE, ''))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = piece.decode('utf-8')
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")



def get_score_dev(out_test,test_examples, text_predictions,na_probs):
  em_scores_raw = {}
  f1_scores_raw = {}
  qas_to_has_ans = {}
  for (index_example, example) in enumerate(test_examples):
    if example.qas_id in text_predictions:
      qas_to_has_ans[example.qas_id] =  (example.start_position >= 0)
      f1_scores_raw[example.qas_id] = compute_f1(example.orig_answer_text, text_predictions[example.qas_id])
      em_scores_raw[example.qas_id] = compute_em(example.orig_answer_text, text_predictions[example.qas_id])
  
  best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(text_predictions, em_scores_raw, na_probs, qas_to_has_ans)
  best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(text_predictions, f1_scores_raw, na_probs, qas_to_has_ans)
  out_test['best_exact'] = best_exact
  out_test['exact_thresh'] = exact_thresh
  out_test['has_ans_exact'] = has_ans_exact
  out_test['best_f1'] = best_f1
  out_test['f1_thresh'] = f1_thresh
  out_test['has_ans_f1'] = has_ans_f1
  # return f1_scores, em_scores
def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
  # (all_predictions,f1_raw,
  #                                  scores_diff_json, qid_to_has_ans)


# preds, f1_raw, na_probs, qid_to_has_ans)
# best_f1, f1_thresh, has_ans_f1
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]

  has_ans_score, has_ans_cnt = 0, 0
  for qid in qid_list:
    if not qid_to_has_ans[qid]: continue
    has_ans_cnt += 1

    if qid not in scores: continue
    has_ans_score += scores[qid]

  return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt



def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1



def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans


def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_em(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  
  return exact_scores, f1_scores


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  
  best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh
  main_eval['has_ans_exact'] = has_ans_exact
  main_eval['has_ans_f1'] = has_ans_f1
