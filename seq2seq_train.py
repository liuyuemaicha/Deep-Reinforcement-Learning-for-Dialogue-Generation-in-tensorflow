# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/pdf/1412.2007v2.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
#from tensorflow.models.rnn.translate import data_utils
#from tensorflow.models.rnn.translate import seq2seq_model
import seq2seq_model
from tensorflow.python.platform import gfile


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 4, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 12, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size",10000, "Query vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "../train_data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../train_data/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("en_cover_dict_path", "../train_data/query_cover_dict", "en_cover_dict_path")
tf.app.flags.DEFINE_string("ff_cover_dict_path", "../train_data/answer_cover_dict", "ff_cover_dict_path")
tf.app.flags.DEFINE_float("fixed_rate", 0.1, "The scala of fixed set in batch train set")
tf.app.flags.DEFINE_string("fixed_set_path", "../train_data/", "")
tf.app.flags.DEFINE_float("weibo_rate", 0.3, "The scala of weibo set in batch train set")
tf.app.flags.DEFINE_string("weibo_set_path", "../train_data/", "")
tf.app.flags.DEFINE_float("qa_rate", 0.0, "The scala of qa set in batch train set")
tf.app.flags.DEFINE_string("qa_set_path", "../train_data", "")
tf.app.flags.DEFINE_boolean("reinforce_learning", True, "Turn on/off reinforce_learning")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]

def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with gfile.GFile(source_path, mode="r") as source_file:
    with gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets): #[bucket_id, (source_size, target_size)]
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, dummy_set,forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size, FLAGS.vocab_size, _buckets, dummy_set,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.fixed_rate, FLAGS.weibo_rate, FLAGS.qa_rate,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model

def train():
  """Train a en->fr translation model using WMT data."""
  #with tf.device("/gpu:0"):
  # Prepare WMT data.
  train_path = os.path.join(FLAGS.data_dir, "chitchat.train")
  fixed_path = os.path.join(FLAGS.data_dir, "chitchat.fixed")
  weibo_path = os.path.join(FLAGS.data_dir, "chitchat.weibo")
  qa_path = os.path.join(FLAGS.data_dir, "chitchat.qa")

  voc_file_path = [train_path+".answer", fixed_path+".answer", weibo_path+".answer", qa_path+".answer",
                     train_path+".query", fixed_path+".query", weibo_path+".query", qa_path+".query"]

  vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.all" % FLAGS.vocab_size)

  data_utils.create_vocabulary(vocab_path, voc_file_path, FLAGS.vocab_size)

  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)


  print("Preparing Chitchat data in %s" % FLAGS.data_dir)
  train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(
      FLAGS.data_dir, vocab, FLAGS.vocab_size)

  print("Preparing Fixed data in %s" % FLAGS.fixed_set_path)
  fixed_path = os.path.join(FLAGS.fixed_set_path, "chitchat.fixed")
  fixed_query , fixed_answer = data_utils.prepare_defined_data(fixed_path, vocab, FLAGS.vocab_size)

  print("Preparing Weibo data in %s" % FLAGS.weibo_set_path)
  weibo_path = os.path.join(FLAGS.weibo_set_path, "chitchat.weibo")
  weibo_query, weibo_answer = data_utils.prepare_defined_data(weibo_path, vocab, FLAGS.vocab_size)

  print("Preparing QA data in %s" % FLAGS.qa_set_path)
  qa_path = os.path.join(FLAGS.qa_set_path, "chitchat.qa")
  qa_query, qa_answer = data_utils.prepare_defined_data(qa_path, vocab, FLAGS.vocab_size)

  dummy_path = os.path.join(FLAGS.data_dir, "chitchat.dummy")
  dummy_set = data_utils.get_dummy_set(dummy_path, vocab, FLAGS.vocab_size)
  print("Get Dummy Set : ", dummy_set)

  with tf.Session() as sess:
#with tf.device("/gpu:1"):
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, dummy_set, False)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)
        dev_set = read_data(dev_query, dev_answer)
        train_set = read_data(train_query, train_answer, FLAGS.max_train_data_size)
        fixed_set = read_data(fixed_query, fixed_answer, FLAGS.max_train_data_size)
        weibo_set = read_data(weibo_query, weibo_answer, FLAGS.max_train_data_size)
        qa_set = read_data(qa_query, qa_answer, FLAGS.max_train_data_size)


        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        en_dict_cover = {}
        fr_dict_cover = {}
        if model.global_step.eval() > FLAGS.steps_per_checkpoint:
            try:
                with open(FLAGS.en_cover_dict_path, "rb") as ef:
                    en_dict_cover = pickle.load(ef)
                    # for line in ef.readlines():
                    #     line = line.strip()
                    #     key, value = line.strip(",")
                    #     en_dict_cover[int(key)]=int(value)
            except Exception:
                print("no find query_cover_file")
            try:
                with open(FLAGS.ff_cover_dict_path, "rb") as ff:
                    fr_dict_cover = pickle.load(ff)
                    # for line in ff.readlines():
                    #     line = line.strip()
                    #     key, value = line.strip(",")
                    #     fr_dict_cover[int(key)]=int(value)
            except Exception:
                print("no find answer_cover_file")

        step_loss_summary = tf.Summary()
        #merge = tf.merge_all_summaries()
        writer = tf.summary.FileWriter("../logs/", sess.graph)

        while True:
          # Choose a bucket according to data distribution. We pick a random number
          # in [0, 1] and use the corresponding interval in train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

          # Get a batch and make a step.
          start_time = time.time()
          encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
              train_set, bucket_id, 0, fixed_set, weibo_set, qa_set)

          if FLAGS.reinforce_learning:
            _, step_loss, _ = model.step_rl(sess, _buckets, encoder_inputs, decoder_inputs, target_weights,
                                            batch_source_encoder, batch_source_decoder, bucket_id)
          else:
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, forward_only=False, force_dec_input=True)

          step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
          loss += step_loss / FLAGS.steps_per_checkpoint
          current_step += 1

          query_size, answer_size = _buckets[bucket_id]
          for batch_index in xrange(FLAGS.batch_size):
            for query_index in xrange(query_size):
              query_word = encoder_inputs[query_index][batch_index]
              if en_dict_cover.has_key(query_word):
                  en_dict_cover[query_word] += 1
              else:
                  en_dict_cover[query_word] = 0

            for answer_index in xrange(answer_size):
              answer_word = decoder_inputs[answer_index][batch_index]
              if fr_dict_cover.has_key(answer_word):
                  fr_dict_cover[answer_word] += 1
              else:
                  fr_dict_cover[answer_word] = 0

          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % FLAGS.steps_per_checkpoint == 0:

            bucket_value = step_loss_summary.value.add()
            bucket_value.tag = "loss"
            bucket_value.simple_value = float(loss)
            writer.add_summary(step_loss_summary, current_step)

            print ("query_dict_cover_num: %s" %(str(en_dict_cover.__len__())))
            print ("answer_dict_cover_num: %s" %(str(fr_dict_cover.__len__())))

            ef = open(FLAGS.en_cover_dict_path, "wb")
            pickle.dump(en_dict_cover, ef)
            ff = open(FLAGS.ff_cover_dict_path, "wb")
            pickle.dump(fr_dict_cover, ff)

            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                             step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "chitchat.model")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
            # Run evals on development set and print their perplexity.
            # for bucket_id in xrange(len(_buckets)):
            #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            #       dev_set, bucket_id)
            #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
            #                                target_weights, bucket_id, True)
            #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            sys.stdout.flush()

def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    pass
    #decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
