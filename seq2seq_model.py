# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from math import log
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

#from tensorflow.models.rnn.translate import data_utils
import data_utils
import seq2seq as rl_seq2seq

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               dummy_set,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               fixed_rate,
               weibo_rate,
               qa_rate,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
			   scope_name='seq2seq',
               dtype=tf.float32):
    
    self.scope_name = scope_name
    with tf.variable_scope(self.scope_name):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.fixed_rate = fixed_rate
        self.weibo_rate = weibo_rate
        self.qa_rate = qa_rate
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.dummy_dialogs = dummy_set

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
          w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
          w = tf.transpose(w_t)
          b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
          output_projection = (w, b)

          def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.cast(
                tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                           num_samples, self.target_vocab_size),
                dtype)
          softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
          single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
          cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return rl_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode,
              dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # for reinforcement learning
        self.force_dec_input = tf.placeholder(tf.bool, name="force_dec_input")
        self.en_output_proj = tf.placeholder(tf.bool, name="en_output_proj")
        # Training outputs and losses.
        #if forward_only:
        self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, tf.select(self.force_dec_input, False, True)),
              softmax_loss_function=softmax_loss_function)
          # If we use output projection, we need to project outputs for decoding.
          #if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
            control_flow_ops.cond(
              self.en_output_proj,
              lambda: tf.matmul(output, output_projection[0]) + output_projection[1],
              lambda: output
            )
            for output in self.outputs[b]
          ]

        # Gradients and SGD update operation for training the model.
        self.tvars = tf.trainable_variables()
        #if not forward_only:
        self.gradient_norms = []
        self.updates = []
        self.advantage = [tf.placeholder(tf.float32, name="advantage_%i" % i) for i in range(len(buckets))]
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        for b in xrange(len(buckets)):
            adjusted_losses = tf.sub(self.losses[b], self.advantage[b])
            gradients = tf.gradients(adjusted_losses, self.tvars)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, self.tvars), global_step=self.global_step))

        # self.saver = tf.train.Saver(tf.all_variables())
        all_variables = [k for k in tf.global_variables() if k.name.startswith(self.scope_name)]
        self.saver = tf.train.Saver(all_variables)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only=True, force_dec_input=False, advantage=None, debug=True):
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {
      self.force_dec_input.name:  force_dec_input,
      self.en_output_proj:        (forward_only),
    }
    for l in xrange(len(self.buckets)):
      input_feed[self.advantage[l].name] = advantage[l] if advantage else 0
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only: # normal training
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else: # testing or reinforcement learning
      output_feed = [self.encoder_state[bucket_id], self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.


  def step_rl(self, session, buckets, encoder_inputs, decoder_inputs, target_weights, batch_source_encoder,
              batch_source_decoder, bucket_id, rev_vocab=None, debug=True):

    # initialize
    init_inputs = [encoder_inputs, decoder_inputs, target_weights, bucket_id]

    batch_mask = [1 for _ in xrange(self.batch_size)]

    # debug
    # resp_tokens, resp_txt = self.logits2tokens(encoder_inputs, rev_vocab, sent_max_length, reverse=True)
    # if debug: print("[INPUT]:", resp_tokens)

    # Initialize
    #if debug: print("step_rf INPUTS: %s" %encoder_inputs)
    #if debug: print("step_rf TARGET: %s" %decoder_inputs)
    #if debug: print("batch_source_encoder: %s" %batch_source_encoder)
    #if debug: print("batch_source_decoder: %s" %batch_source_decoder)

    ep_rewards, ep_step_loss, enc_states = [], [], []
    ep_encoder_inputs, ep_target_weights, ep_bucket_id = [], [], []
    episode, dialog = 0, []
    # [Episode] per episode = n steps, until break
    print ("ep_encoder_inputs: %s" %ep_encoder_inputs)
    print("bucket: %s, bucekt[-1][0]: %s" %(self.buckets, self.buckets[-1][0]))
    while True:
      #----[Step]------with general function-----------------------------

      #ep_encoder_inputs.append(self.remove_type(encoder_inputs, type=0))
      ep_encoder_inputs.append(batch_source_encoder)
      #decoder_len = [len(seq) for seq in batch_source_decoder]

      if debug: print ("ep_encoder_inputs shape: ", np.shape(ep_encoder_inputs))
      #if debug: print ("[INPUT]: %s" %ep_encoder_inputs[-1])

      encoder_state, step_loss, output_logits = self.step(session, encoder_inputs, decoder_inputs, target_weights,
                          bucket_id, forward_only=True, force_dec_input=False)
      print("encoder_state: " , np.shape(encoder_state))
      ep_target_weights.append(target_weights)
      ep_bucket_id.append(bucket_id)
      ep_step_loss.append(step_loss)

      state_tran = np.transpose(encoder_state, axes=(1,0,2))
      print("state_tran: ", np.shape(state_tran))
      state_vec = np.reshape(state_tran, (self.batch_size, -1))
      print("state_vec: ", np.shape(state_vec))
      enc_states.append(state_vec)

      resp_tokens = self.remove_type(output_logits, self.buckets[bucket_id], type=1)
      #print("remove_type resps: %s" %resp_tokens)
      #if debug: print("[RESP]: (%.4f), resp len: %s, content: %s" %(step_loss, len(resp_tokens), resp_tokens))
      try:
        encoder_trans = np.transpose(ep_encoder_inputs, axes=(1,0))
      except ValueError:
        encoder_trans = np.transpose(ep_encoder_inputs, axes=(1,0,2))
      #if debug: print ("[ep_encoder_inputs] shape: ", np.shape(ep_encoder_inputs))
      if debug: print ("[encoder_trans] shape: ", np.shape(encoder_trans))
      #if episode == 5: print (2/0)

      for i, (resp, ep_encoder) in enumerate(zip(resp_tokens, encoder_trans)):
          print("resp: ", resp)

          if (len(resp) <= 3) or (resp in self.dummy_dialogs) or (resp in ep_encoder.tolist()):
              batch_mask[i] = 0
              print("make mask index: %d, batch_mask: %s" %(i, batch_mask))

      if sum(batch_mask)==0 or episode>5: break

      #----[Reward]----------------------------------------
      # r1: Ease of answering
      print("r1: Ease of answering")
      r1 = [self.logProb(session, buckets, resp_tokens, [d for _ in resp_tokens], mask= batch_mask) for d in self.dummy_dialogs]
      print("r1: final value: ", r1)
      r1 = -np.mean(r1) if r1 else 0

      # r2: Information Flow
      r2_list = []
      if len(enc_states) < 2:
        r2 = 0
      else:
        batch_vec_a, batch_vec_b = enc_states[-2], enc_states[-1]
        for i, (vec_a, vec_b) in enumerate(zip(batch_vec_a, batch_vec_b)):
          if batch_mask[i] == 0: continue
          rr2 = sum(vec_a*vec_b) / sum(abs(vec_a)*abs(vec_b))
          #print("vec_a*vec_b: %s" %sum(vec_a*vec_b))
          #print("r2: %s" %r2)
          if(rr2 < 0):
            print("rr2: ", rr2)
            print("vec_a: ", vec_a)
            print("vec_b: ", vec_b)
            rr2 = -rr2
          else:
            rr2 = -log(rr2)
          r2_list.append(rr2)
        r2 = sum(r2_list)/len(r2_list)
      # r3: Semantic Coherence
      print("r3: Semantic Coherence")
      r3 = -self.logProb(session, buckets, resp_tokens, ep_encoder_inputs[-1], mask= batch_mask)

      # Episode total reward
      print("r1: %s, r2: %s, r3: %s" %(r1,r2,r3))
      R = 0.25*r1 + 0.25*r2 + 0.5*r3
      ep_rewards.append(R)
      #----------------------------------------------------
      episode += 1

      #prepare for next dialogue
      bk_id = []
      for i in range(len(resp_tokens)):
        bk_id.append(min([b for b in range(len(buckets)) if buckets[b][0] >= len(resp_tokens[i])]))
      bucket_id = max(bk_id)
      feed_data = {bucket_id: [(resp_tokens, [])]}
      encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, _ = self.get_batch(feed_data, bucket_id, type=2)

    if len(ep_rewards) == 0: ep_rewards.append(0)
    print("[Step] final:", episode, ep_rewards)
    # gradient decent according to batch rewards
    rto = 0.0
    if (len(ep_step_loss) <= 1) or (len(ep_rewards) <=1) or (max(ep_rewards) - min(ep_rewards) == 0):
        rto = 0.0
    else:
        rto = (max(ep_step_loss) - min(ep_step_loss)) / (max(ep_rewards) - min(ep_rewards))
    advantage = [np.mean(ep_rewards)*rto] * len(buckets)
    print("advantage: %s" %advantage)
    _, step_loss, _ = self.step(session, init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3],
              forward_only=False, force_dec_input=False, advantage=advantage)

    return None, step_loss, None


# log(P(b|a)), the conditional likelyhood
  def logProb(self, session, buckets, tokens_a, tokens_b, mask=None):
    def softmax(x):
      return np.exp(x) / np.sum(np.exp(x), axis=0)

    # prepare for next dialogue
    #bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(tokens_a) and buckets[b][1] > len(tokens_b)])
    #print("tokens_a: %s" %tokens_a)
    print("tokens_b: %s" %tokens_b)

    bk_id = []
    for i in xrange(len(tokens_a)):
        bk_id.append(min([b for b in xrange(len(buckets))
                          if buckets[b][0] >= len(tokens_a[i]) and buckets[b][1] >= len(tokens_b[i])]))
    bucket_id = max(bk_id)


    #print("bucket_id: %s" %bucket_id)

    feed_data = {bucket_id: zip(tokens_a, tokens_b)}

    #print("logProb feed_back: %s" %feed_data[bucket_id])
    encoder_inputs, decoder_inputs, target_weights, _, _ = self.get_batch(feed_data, bucket_id, type=1)
    #print("logProb: encoder: %s; decoder: %s" %(encoder_inputs, decoder_inputs))
    # step
    _, _, output_logits = self.step(session, encoder_inputs, decoder_inputs, target_weights,
                        bucket_id, forward_only=True, force_dec_input=True)

    logits_t = np.transpose(output_logits, (1,0,2))
    print("logits_t shape: " ,np.shape(logits_t))


    sum_p = []
    for i, (tokens, logits) in enumerate(zip(tokens_b, logits_t)):
        print("tokens: %s, index: %d" %(tokens, i))
        #print("logits: %s" %logits)

        #if np.sum(tokens) == 0: break
        if mask[i] == 0: continue
        p = 1
        for t, logit in zip(tokens, logits):
            #print("logProb: logit: %s" %logit)
            norm = softmax(logit)[t]
            #print ("t: %s, norm: %s" %(t, norm))
            p *= norm
        if p < 1e-100:
          #print ("p: ", p)
          p = 1e-100
        p = log(p) / len(tokens)
        print ("logProb: p: %s" %(p))
        sum_p.append(p)
    re = np.sum(sum_p)/len(sum_p)
    #print("logProb: P: %s" %(re))
    return re

  def remove_type(self, sequence, bucket,type=0):
      tokens = []
      resps = []
      if type == 0:
        tokens = [i for i in [t for t in reversed(sequence)] if i.sum() != 0]
      elif type == 1:
        #print ("remove_type type=1 tokens: %s" %sequence)

        for seq in sequence:
             #print("seq: %s" %seq)
             token = []
             for t in seq:
                 #print("seq_t: %s" %t)
                 # t = list(t)
                 # print("list(t): %s" %t)
                 # t = np.array(t)
                 # print("array(t): %s" %t)
                 token.append(int(np.argmax(t, axis=0)))
             tokens.append(token)

        #tokens = [i for i in [int(np.argmax(t, axis=1)) for t in [seq for seq in sequence]]]
        #tokens = [i for i in [int(t.index(max(t))) for t in [seq for seq in sequence]]]
      else:
        print ("type only 0(encoder_inputs) or 1(decoder_outputs)")
      #print("remove_type tokens: %s" %tokens)
      tokens_t = []
      for col in range(len(tokens[0])):
            tokens_t.append([tokens[row][col] for row in range(len(tokens))])

      for seq in tokens_t:
            if data_utils.EOS_ID in seq:
                resps.append(seq[:seq.index(data_utils.EOS_ID)][:bucket[1]])
            else:
                resps.append(seq[:bucket[1]])
      return resps

  # make logits to tokens
  def logits2tokens(self, logits, rev_vocab, sent_max_length=None, reverse=False):
    if reverse:
      tokens = [t[0] for t in reversed(logits)]
    else:
      tokens = [int(np.argmax(t, axis=1)) for t in logits]
    if data_utils.EOS_ID in tokens:
      eos = tokens.index(data_utils.EOS_ID)
      tokens = tokens[:eos]
    txt = [rev_vocab[t] for t in tokens]
    if sent_max_length:
      tokens, txt = tokens[:sent_max_length], txt[:sent_max_length]
    return tokens, txt


  def discount_rewards(self, r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


  def get_batch(self, train_data, bucket_id, type=0, fixed_set=None, weibo_set=None, qa_set=None):

    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    fixed_train_num = 0
    weibo_train_num = 0
    qa_train_num = 0
    if type == 0:
      fixed_num = int(self.batch_size * self.fixed_rate)
      fixed_len = len(fixed_set[bucket_id])
      fixed_train_num = min(fixed_num, fixed_len)

      weibo_num = int(self.batch_size * self.weibo_rate)
      weibo_len = len(weibo_set[bucket_id])
      weibo_train_num = min(weibo_num, weibo_len) + fixed_train_num

      qa_num = int(self.batch_size * self.qa_rate)
      qa_len = len(qa_set[bucket_id])
      qa_train_num = min(qa_num, qa_len) + weibo_train_num

    #print("Batch_Size: %s" %self.batch_size)
    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_source_encoder, batch_source_decoder = [], []
    #print("bucket_id: %s" %bucket_id)
    for batch_i in xrange(self.batch_size):
      if type == 1:
        encoder_input, decoder_input = train_data[bucket_id][batch_i]
      elif type == 2:
        #print("data[bucket_id]: ", data[bucket_id][0])
        encoder_input_a, decoder_input = train_data[bucket_id][0]
        encoder_input = encoder_input_a[batch_i]
      elif type == 0:
        if fixed_train_num > batch_i:
          encoder_input, decoder_input = random.choice(fixed_set[bucket_id])
          print("fixed en: %s, de: %s" %(encoder_input, decoder_input))
        elif weibo_train_num > batch_i:
          encoder_input, decoder_input = random.choice(weibo_set[bucket_id])
          print("weibo en: %s, de: %s" %(encoder_input, decoder_input))
        elif qa_train_num > batch_i:
          encoder_input, decoder_input = random.choice(qa_set[bucket_id])
          print("qa en: %s, de: %s" %(encoder_input, decoder_input))
        else:
          encoder_input, decoder_input = random.choice(train_data[bucket_id])
          print("train en: %s, de: %s" %(encoder_input, decoder_input))

      batch_source_encoder.append(encoder_input)
      batch_source_decoder.append(decoder_input)
      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder
