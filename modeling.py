from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# from tf.keras.callbacks import TensorBoard

# import model_utils

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


class XLNetMainLayer(tf.keras.layers.Layer):
  def __init__(self,config,name='transformer',**kwargs):
    super(XLNetMainLayer, self).__init__(name=name, **kwargs)
    self.n_token = config.n_token
    self.n_layer = config.n_layer
    self.d_model = config.d_model
    self.n_head = config.n_head
    self.d_head = config.d_head
    self.d_inner = config.d_inner
    self.dropout = config.dropout
    self.dropatt = config.dropatt
    self.attn_type  = config.attn_type
    self.clamp_len = config.clamp_len
    self.initializer = config.initializer
    self.ff_activation= config.ff_activation #'relu'
    self.bi_data = config.bi_data #False
    self.mem_len = config.mem_len #None
    self.same_length =config.same_length #False
    self.reuse_len = config.reuse_len #None
    self.init_checkpoint = config.init_checkpoint
    self.word_embedding = WordEmbedding(config,name="word_embedding")
    self.word_embedding_mask = WordEmbeddingMask(config, name="mask_emb")
    self.layer = [XLNetLayer(config,name='layer_{}'.format(i)) for i in range(self.n_layer)]  
    self.drop_out = tf.keras.layers.Dropout(self.dropout)
 
  def create_mask(self,qlen, mlen, dtype=tf.float32, same_length=False):
    """create causal attention mask."""
    attn_mask = tf.ones([qlen, qlen], dtype=dtype)
    mask_u = tf.linalg.band_part(attn_mask, 0, -1)
    mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
      mask_l = tf.linalg.band_part(attn_mask, -1, 0)
      ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

    return ret
  def relative_positional_encoding(self,qlen,klen,bsz=None,dtype=None):
    freq_seq = tf.range(0, self.d_model, 2.0)
    if dtype is not None and dtype != tf.float32:
      freq_seq = tf.cast(freq_seq, dtype=dtype)
    inv_freq = 1 / (10000 ** (freq_seq / self.d_model))

    if self.attn_type == 'bi':
      # beg, end = klen - 1, -qlen
      beg, end = klen, -qlen
    elif self.attn_type == 'uni':
      # beg, end = klen - 1, -1
      beg, end = klen, -1
    else:
      raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

    if self.bi_data:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      bwd_pos_seq = tf.range(-beg, -end, 1.0)

      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
        bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)

      if self.clamp_len > 0:
        fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
        bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)

      if bsz is not None:
        fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
        bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
      else:
        fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
        bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

      pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      if dtype is not None and dtype != tf.float32:
        fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
      if self.clamp_len > 0:
        fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
      pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
    return pos_emb

  def positional_embedding(self,pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    pos_emb = pos_emb[:, None, :]

    if bsz is not None:
      pos_emb = tf.tile(pos_emb, [1, bsz, 1])

    return pos_emb
    
  def cache_mem(self,curr_out,prev_mem):
    """cache hidden states into memory."""
    if self.mem_len is None or self.mem_len == 0:
      return None
    else:
      if self.reuse_len is not None and self.reuse_len > 0:
        curr_out = curr_out[:self.reuse_len]

      if prev_mem is None:
        new_mem = curr_out[-self.mem_len:]
      else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-self.mem_len:]
    return tf.stop_gradient(new_mem)

  def build(self,input_shape):
    self.r_w_bias = self.add_weight('r_w_bias',shape=[self.n_layer,self.n_head,self.d_head],initializer=self.initializer)
    self.r_r_bias = self.add_weight('r_r_bias',shape=[self.n_layer,self.n_head,self.d_head],initializer=self.initializer)
    self.r_s_bias = self.add_weight('r_s_bias',shape=[self.n_layer,self.n_head,self.d_head],initializer=self.initializer)
    self.seg_embed = self.add_weight('seg_embed',shape=[self.n_layer,2,self.n_head,self.d_head],initializer=self.initializer)

    super().build(input_shape)
    
  def call(self,inputs,training=False):
    inp_k, inp_q, mems, input_mask, perm_mask, seg_id, target_mapping = inputs
    bsz = tf.shape(input=inp_k)[1]
    qlen = tf.shape(input=inp_k)[0]
    mlen = tf.shape(input=mems[0])[0] if mems is not None else 0
    klen = mlen + qlen
    tf_float = tf.float32


    if self.attn_type == 'uni':
      attn_mask = self.create_mask(qlen, mlen, tf_float, self.same_length)
      attn_mask = attn_mask[:, :, None, None]
    elif self.attn_type == 'bi':
      attn_mask = None
    else:
      raise ValueError('Unsupported attention type: {}'.format(attn_type))

    # data mask: input mask & perm mask
    if input_mask is not None and perm_mask is not None:
      data_mask = input_mask[None] + perm_mask
    elif input_mask is not None and perm_mask is None:
      data_mask = input_mask[None]
    elif input_mask is None and perm_mask is not None:
      data_mask = perm_mask
    else:
      data_mask = None

    if data_mask is not None:
      # all mems can be attended to
      mems_mask = tf.zeros([tf.shape(input=data_mask)[0], mlen, bsz],
                          dtype=tf_float)
      data_mask = tf.concat([mems_mask, data_mask], 1)
      if attn_mask is None:
        attn_mask = data_mask[:, :, :, None]
      else:
        attn_mask += data_mask[:, :, :, None]

    if attn_mask is not None:
      attn_mask = tf.cast(attn_mask > 0, dtype=tf_float)

    if attn_mask is not None:
      non_tgt_mask = -tf.eye(qlen, dtype=tf_float)
      non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=tf_float),
                                non_tgt_mask], axis=-1)
      non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                            dtype=tf_float)
    else:
      non_tgt_mask = None


    word_emb_k,lookup_table = self.word_embedding(inp_k)
    if inp_q is not None:
      word_emb_q =self.word_embedding_mask(inp_q,target_mapping,word_emb_k,bsz)
    output_h = self.drop_out(word_emb_k,training=training)
    if inp_q is not None:
      output_g = self.drop_out(word_emb_q,training=training)
    else:
      output_g =None
    if seg_id is not None:
      mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)
      cat_ids = tf.concat([mem_pad, seg_id], 0)
      seg_mat = tf.cast(
          tf.logical_not(tf.equal(seg_id[:, None], cat_ids[None, :])),
          tf.int32)
      seg_mat = tf.one_hot(seg_mat, 2, dtype=tf_float)
    else:
      seg_mat = None
    pos_emb = self.relative_positional_encoding(qlen, klen,bsz=bsz,dtype=tf_float)
    pos_emb =self.drop_out(pos_emb,training = training)
    if mems is None:
      mems = [None] * self.n_layer
    new_mems = []
    for i in range(self.n_layer):
      new_mems.append(self.cache_mem(output_h,mems[i]))
      if inp_q is not None:
        output_h, output_g = self.layer[i](
          r_w_bias = self.r_w_bias[i],
          r_s_bias = self.r_s_bias[i],
          r_r_bias = self.r_r_bias[i],
          seg_embed = self.seg_embed[i],
          h = output_h,
          g = output_g,
          r = pos_emb,
          mems =mems[i],
          seg_mat=seg_mat,
          attn_mask_h = non_tgt_mask,
          attn_mask_g = attn_mask,
          target_mapping =target_mapping,
          training =training)
      else:
        output_h = self.layer[i](
          r_w_bias = self.r_w_bias[i],
          r_s_bias = self.r_s_bias[i],
          r_r_bias = self.r_r_bias[i],
          seg_embed = self.seg_embed[i],
          h = output_h,
          g = output_g,
          r = pos_emb,
          mems =mems[i],
          seg_mat=seg_mat,
          attn_mask_h = non_tgt_mask,
          attn_mask_g = attn_mask,
          target_mapping =target_mapping,
          training =training)
    if inp_q is not None:
      output =self.drop_out(output_g,training=training)
    else:
      output =self.drop_out(output_h,training=training)
    # self.assign_checkpoint_to_weights()

    return output,new_mems, lookup_table


class RelativeAttention(tf.keras.layers.Layer):
  def __init__(self,config,name='rel_attn',**kwargs):
    super(RelativeAttention, self).__init__(name=name, **kwargs)

    self.n_head = config.n_head
    self.d_head = config.d_head
    self.d_model = config.d_model
    self.scale = 1 / (self.d_head ** 0.5)
    self.initializer = config.initializer
    self.layer_norm = tf.keras.layers.LayerNormalization(name="LayerNorm")
    self.drop_out = tf.keras.layers.Dropout(config.dropout)
    
  def build(self, input_shape):
    self.k = self.add_weight('k/kernel', shape=[self.d_model,self.n_head,self.d_head],initializer=self.initializer)
    self.o = self.add_weight('o/kernel', shape=[self.d_model,self.n_head,self.d_head],initializer=self.initializer)
    self.q = self.add_weight('q/kernel', shape=[self.d_model,self.n_head,self.d_head],initializer=self.initializer)
    self.r = self.add_weight('r/kernel', shape=[self.d_model,self.n_head,self.d_head],initializer=self.initializer)
    self.v = self.add_weight('v/kernel', shape=[self.d_model,self.n_head,self.d_head],initializer=self.initializer)
        
    super().build(input_shape)
    
  def call(self,r_w_bias,r_s_bias,r_r_bias,seg_embed,h,g,r,mems,seg_mat,attn_mask_h,attn_mask_g,target_mapping,training=False):
    if g is not None:
      # two stream attention
      if mems is not None and mems.shape.ndims > 1:
        cat = tf.concat([mems, h], 0)
      else:
        cat = h
      k_head_h = tf.einsum('ibh,hnd->ibnd', cat,self.k)
      v_head_h = tf.einsum('ibh,hnd->ibnd', cat,self.v)
      k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r)
      q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q)
      attn_vec_h = self.rel_attn_core(
        r_w_bias, r_s_bias, r_r_bias, seg_embed, q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, training=training)
      output_h = self.post_attention(h, attn_vec_h, training=training)
      q_head_g = tf.einsum('ibh,hnd->ibnd',g, self.q)
      if target_mapping is not None:
        q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
        attn_vec_g = self.rel_attn_core(
          r_w_bias,r_s_bias,r_r_bias,seg_embed,q_head_g, k_head_h, v_head_h, k_head_r,seg_mat, attn_mask_g, training=training)
        attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
      else:
        attn_vec_g = self.rel_attn_core(
          r_w_bias,r_s_bias,r_r_bias,seg_embed,q_head_g, k_head_h, v_head_h, k_head_r,seg_mat, attn_mask_g, training=training)
      
      output_g = self.post_attention(g, attn_vec_g, training=training)
      return output_h, output_g
    else:
      # Nomal relarive attetion when fine-tunning
      if mems is not None and mems.shape.ndims > 1:
        cat = tf.concat([mems, h], 0)
      else:
        cat = h
      q_head_h = tf.einsum('ibh,hnd->ibnd',h,self.q)
      k_head_h = tf.einsum('ibh,hnd->ibnd',cat,self.k)
      v_head_h = tf.einsum('ibh,hnd->ibnd',cat,self.v)
      k_head_r = tf.einsum('ibh,hnd->ibnd',r,self.r)
      attn_vec = self.rel_attn_core(
        r_w_bias,r_s_bias,r_r_bias,seg_embed, q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, training=training)
      output = self.post_attention(h, attn_vec, training=training)
      return output


  def post_attention(self,h, attn_vec, training=False, residual=True):
    attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.o)
    attn_out = self.drop_out(attn_out, training=training)
    if residual:
      output = self.layer_norm(attn_out+h)
    else:
      output = self.layer_norm(attn_out)
    return output


  def rel_attn_core(self,r_w_bias,r_s_bias,r_r_bias,seg_embed,q_head, k_head_h, v_head_h, k_head_r,seg_mat, attn_mask, training=False):
    ac = tf.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)
    bd = tf.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
    bd = self.rel_shift(bd, klen=tf.shape(input=ac)[1])
    # segment based attention score
    if seg_mat is None:
      ef = 0
    else:
      ef = tf.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
      ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)
    # merge attention scores and perform masking
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
      # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
      attn_score = attn_score - 1e30 * attn_mask
    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = self.drop_out(attn_prob, training=training)

    # attention output
    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
    return attn_vec

  def rel_shift(self,x, klen=-1):
    """perform relative shift to form the relative attention score."""
    x_size = tf.shape(input=x)

    x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

    return x
class XLNetLayer(tf.keras.layers.Layer):
  def __init__(self,config,name="layer",**kwargs):
    super(XLNetLayer, self).__init__(name=name, **kwargs)
    self.rel_attn = RelativeAttention(config, name ='rel_attn')
    self.pos_ffn = PossitionwiseFFN(config,name='ff')
    self.drop_out = tf.keras.layers.Dropout(config.dropout)
  def call(self,r_w_bias,r_s_bias,r_r_bias,seg_embed,h,g,r,mems,seg_mat,attn_mask_h,attn_mask_g,target_mapping,training=False):
    if g is not None:
      output_h,output_g = self.rel_attn(r_w_bias,r_s_bias,r_r_bias,seg_embed,h,g,r,mems,seg_mat,attn_mask_h,attn_mask_g,target_mapping,training)
    else:
      output_h = self.rel_attn(r_w_bias,r_s_bias,r_r_bias,seg_embed,h,g,r,mems,seg_mat,attn_mask_h,attn_mask_g,target_mapping,training)
    output_h = self.pos_ffn(output_h,training)
    if g is not None:
      output_g = self.pos_ffn(output_g,training)
      return output_h,output_g
    else:
      return output_h
class PossitionwiseFFN(tf.keras.layers.Layer):
  def __init__(self,config,name='ff',**kwargs):
    super(PossitionwiseFFN, self).__init__(name=name, **kwargs)
    self.activation = tf.nn.relu if config.ff_activation == 'relu' else gelu 
    self.layer_norm = tf.keras.layers.LayerNormalization(name="LayerNorm")
    self.drop_out = tf.keras.layers.Dropout(config.dropout)
    self.layer_1 = tf.keras.layers.Dense(config.d_inner, kernel_initializer = config.initializer,name ="layer_1", activation= self.activation)
    self.layer_2 = tf.keras.layers.Dense(config.d_model, kernel_initializer = config.initializer,name ="layer_2")
  def call(self,inp,training=False):
    output = inp
    output = self.layer_1(output)
    output = self.drop_out(output, training = training)
    output = self.layer_2(output)
    output = self.drop_out(output, training = training)
    output = self.layer_norm(output + inp)
    return output
class XLNetModelQA(tf.keras.Model):
  def __init__(self,config,name="model",**kwargs):
    super(XLNetModelQA, self).__init__(name=name, **kwargs)
    self.n_layer =config.n_layer
    self.d_model =config.d_model
    self.n_best_size = config.n_best_size
    self.start_n_top = config.start_n_top
    self.end_n_top = config.end_n_top
    self.transformer = XLNetMainLayer(config,name="transformer")
    self.dense_start = tf.keras.layers.Dense(1, kernel_initializer = config.initializer,name ="dense_start")
    self.dense_end_1 = tf.keras.layers.Dense(config.d_model, kernel_initializer = config.initializer,name ="dense_end_1", activation='tanh')
    self.dense_end_2 = tf.keras.layers.Dense(1, kernel_initializer = config.initializer,name ="dense_end_2")
    self.layer_norm = tf.keras.layers.LayerNormalization(name="LayerNorm")
    self.dense_answer_1 = tf.keras.layers.Dense(config.d_model, kernel_initializer = config.initializer,name ="dense_answer_1" ,activation='tanh')
    self.dense_answer_2 = tf.keras.layers.Dense(1, kernel_initializer = config.initializer,name ="dense_answer_2",use_bias= False )
    self.drop_out = tf.keras.layers.Dropout(config.dropout)
    self.layer_norm_not_train = tf.keras.layers.LayerNormalization(trainable=False)
    
  def call(self,inputs,training=False,mode="test"):
    if not training and mode=="train":
      raise ValueError('When Evaluate must test')
    if mode !="train" and mode !="test":
      raise ValueError('Mode must be train or test')
    inp_k = tf.transpose(inputs['input_ids'])
    input_mask =  tf.transpose(inputs['input_mask'])
    p_mask =  inputs['p_mask']
    cls_index = tf.reshape(inputs['cls_index'],[-1])
    if training and mode=="train":
      start_positions = tf.reshape(inputs["start_positions"], [-1])
    seg_id = tf.transpose(inputs['segment_ids'])
    inp_q = None
    mems = None
    perm_mask = None
    target_mapping =None
    seq_len = tf.shape(input=inp_k)[0]
    bsz = tf.shape(input=inp_k)[1]
    return_dict = {}
    if mode == "test":
      return_dict['unique_ids'] = tf.transpose(inputs['unique_ids'])
    if mode =="test":
      output_transformer = self.transformer((inp_k, inp_q, mems, input_mask, perm_mask, seg_id, target_mapping),training=False)[0]
    else:
      output_transformer = self.transformer((inp_k, inp_q, mems, input_mask, perm_mask, seg_id, target_mapping),training=True)[0]
    
    
    start_logits = self.dense_start(output_transformer)
    start_logits = tf.transpose(a=tf.squeeze(start_logits, -1), perm=[1, 0])
    # print(start_logits.shape,"bbb")
    # print((1 - p_mask).shape)
    start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
    start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)
    return_dict["start_log_probs"] = start_log_probs
    if training:
      # end_logits = self.dense_end(output_transformer)
      # end_logits = tf.transpose(a=tf.squeeze(end_logits, -1), perm=[1, 0])
      # # print(start_logits.shape,"bbb")
      # # print((1 - p_mask).shape)
      # # start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
      # end_log_probs = tf.nn.log_softmax(end_logits, -1)
      # return_dict["end_log_probs"] = end_log_probs
      if mode == "train":
        start_index = tf.one_hot(start_positions, depth=seq_len, axis=-1,
                                dtype=tf.float32)
        start_features = tf.einsum("lbh,bl->bh", output_transformer, start_index)
        start_features = tf.tile(start_features[None], [seq_len, 1, 1])  
        # print(tf.concat([output_transformer, start_features], axis=-1).shape,"ccc")
      else:
        start_features =tf.zeros(shape=(seq_len,bsz,self.d_model))
      end_logits = self.dense_end_1(tf.concat([output_transformer, start_features], axis=-1))

      end_logits = self.layer_norm(end_logits)
      end_logits = self.dense_end_2(end_logits)
      end_logits = tf.transpose(a=tf.squeeze(end_logits, -1), perm=[1, 0])
      end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)              
      # print(end_log_probs.shape,"cccccccccccc")
      return_dict["end_log_probs"] = end_log_probs
      
      #   end_logits_dev = self.dense_end_2(output_transformer)
      #   end_logits_dev = tf.transpose(a=tf.squeeze(end_logits_dev, -1), perm=[1, 0])
      #   # print(start_logits.shape,"bbb")
      #   # print((1 - p_mask).shape)
      #   end_logits_masked_dev = end_logits_dev * (1 - p_mask) - 1e30 * p_mask
      #   end_log_probs_dev = tf.nn.log_softmax(end_logits_masked_dev, -1)
      #   return_dict["end_log_probs"] = end_log_probs_dev
      # print(end_log_probs.shape,"vvvvvvvvvv")
    else:
      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, self.start_n_top)
      start_index = tf.one_hot(start_top_index,
                              depth=seq_len, axis=-1, dtype=tf.float32)
      start_features = tf.einsum("lbh,bkl->bkh", output_transformer, start_index)
      end_input = tf.tile(output_transformer[:, :, None],
                          [1, 1, self.start_n_top, 1])
      start_features = tf.tile(start_features[None],
                              [seq_len, 1, 1, 1])
      end_input = tf.concat([end_input, start_features], axis=-1)
      end_logits = self.dense_end_1(end_input)
      # print(end_logits.shape)
      # print(self.layer_norm.shape)
      end_logits = self.layer_norm_not_train(end_logits)
      end_logits = self.dense_end_2(end_logits)
      end_logits = tf.reshape(end_logits, [seq_len, -1, self.start_n_top])
      end_logits = tf.transpose(a=end_logits, perm=[1, 2, 0])
      end_logits_masked = end_logits * (
          1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
      end_top_log_probs, end_top_index = tf.nn.top_k(
          end_log_probs, k=self.end_n_top)
      end_top_log_probs = tf.reshape(
          end_top_log_probs,
          [-1, self.start_n_top * self.end_n_top])
      end_top_index = tf.reshape(
          end_top_index,
          [-1, self.start_n_top * self.end_n_top])
      
      return_dict["start_top_log_probs"] = start_top_log_probs
      return_dict["start_top_index"] = start_top_index
      return_dict["end_top_log_probs"] = end_top_log_probs
      return_dict["end_top_index"] = end_top_index
    # get the representation of CLS
    cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
    # print(cls_index.shape,"nnnnn")
    cls_feature = tf.einsum("lbh,bl->bh", output_transformer, cls_index)

    # get the representation of START
    start_p = tf.nn.softmax(start_logits_masked, axis=-1,
                            name="softmax_start")
    start_feature = tf.einsum("lbh,bl->bh", output_transformer, start_p)

    # note(zhiliny): no dependency on end_feature so that we can obtain
    # one single `cls_logits` for each sample
    ans_feature = tf.concat([start_feature, cls_feature], -1)
    ans_feature = self.dense_answer_1(ans_feature)
    if mode == "test":
      ans_feature = self.drop_out(ans_feature,training=False)
    else:
      ans_feature = self.drop_out(ans_feature,training=True)

    cls_logits = self.dense_answer_2(ans_feature)
    cls_logits = tf.squeeze(cls_logits)
    return_dict["cls_logits"] = cls_logits
    return return_dict


class WordEmbedding(tf.keras.layers.Layer):
  def __init__(self,config,name='word_embedding',**kwargs):
    super(WordEmbedding, self).__init__(name=name, **kwargs)
    self.n_token = config.n_token
    self.d_embed = config.d_model
    self.initializer = config.initializer

    

  def build(self, input_shape):
    
    self.lookup_table = self.add_weight('lookup_table',shape=[self.n_token,self.d_embed],initializer=self.initializer)
    super().build(input_shape)
  def call(self,inputs):
    return tf.nn.embedding_lookup(params=self.lookup_table, ids=inputs), self.lookup_table



class WordEmbeddingMask(tf.keras.layers.Layer):
  def __init__(self,config,name='mask_emb',**kwargs):
    super(WordEmbeddingMask, self).__init__(name=name, **kwargs)
    self.d_embed = config.d_model
    
    
  def build(self,input_shape):
    self.mask_emb = self.add_weight('mask_emb',shape=[1,1,self.d_embed])

    super().build(input_shape)

  def call(self,inp_q,target_mapping,word_emb_k,bsz):
    if target_mapping is not None:
      word_emb_q = tf.tile(self.mask_emb, [tf.shape(input=target_mapping)[0], bsz, 1])
    else:
      inp_q_ext = inp_q[:, :, None]
      word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k

    return word_emb_q
