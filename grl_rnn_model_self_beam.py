import tensorflow as tf
from tensorflow.python.ops import variable_scope
import numpy as np
import grl_seq2seq as rl_seq2seq
import data_utils
import random
from math import log


class grl_model(object):
    def __init__(self, grl_config, name_scope, num_samples=512, forward = False, dtype=tf.float32):
        self.buckets = grl_config.buckets_concat
        self.beam_size = grl_config.beam_size
        self.emb_dim = grl_config.emb_dim
        self.batch_size = grl_config.batch_size
        self.vocab_size = grl_config.vocab_size
        #self.learning_rate = grl_config.learning_rate
        self.learning_rate = tf.Variable(float(grl_config.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * grl_config.learning_rate_decay_factor)
        self.dummy_dialogs = []
        max_gradient_norm = grl_config.max_gradient_norm
        num_layers = grl_config.num_layers

        with tf.name_scope("GRL_Cell"):
            single_cell = tf.nn.rnn_cell.GRUCell(self.emb_dim)
            cells = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        self.global_step = tf.Variable(0, trainable=False)
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(self.buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in xrange(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="deocder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
        self.forward_only = tf.placeholder(tf.bool, name="forward_only")
        self.beam_search = tf.placeholder(tf.bool, name="beam_search")
        self.rewards = [tf.placeholder(tf.float32, name="reward{0}".format(i)) for i in xrange(len(self.buckets))]

        # the top of decoder_inputs is mark
        targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs) - 1)]

        output_projection = None
        softmax_loss_function = None
        if num_samples > 0 and num_samples < self.vocab_size:
            w_t = tf.get_variable("proj_w", [self.vocab_size, self.emb_dim], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast( tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                                           num_samples, self.vocab_size), dtype)
                #return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.vocab_size)

            softmax_loss_function = sampled_loss

        with tf.name_scope("GRL_Seq2Seq"):
            def seq2seq_f(encoder_inputs, decoder_inputs, forward_only, beam_search):
                # return rl_seq2seq.embedding_attention_seq2seq(encoder_inputs=encoder_inputs,
                #                                               decoder_inputs=decoder_inputs,
                #                                               cell= cells,
                #                                               num_encoder_symbols=self.vocab_size,
                #                                               num_decoder_symbols=self.vocab_size,
                #                                               embedding_size=self.emb_dim,
                #                                               output_projection=output_projection,
                #                                               feed_previous=forward_only,
                #                                               beam_size=self.beam_size,
                #                                               dtype=dtype)
                # return rl_seq2seq.embedding_rnn_seq2seq(encoder_inputs=encoder_inputs,
                #                                         decoder_inputs=decoder_inputs,
                #                                         cell=cells,
                #                                         num_encoder_symbols=self.vocab_size,
                #                                         num_decoder_symbols=self.vocab_size,
                #                                         embedding_size=self.emb_dim,
                #                                         output_projection=output_projection,
                #                                         feed_previous=forward_only,
                #                                         beam_search=forward_only,
                #                                         beam_size=self.beam_size,
                #                                         dtype=dtype)
                return rl_seq2seq.embedding_attention_seq2seq(encoder_inputs=encoder_inputs,
                                                              decoder_inputs=decoder_inputs,
                                                              cell=cells,
                                                              num_encoder_symbols=self.vocab_size,
                                                              num_decoder_symbols=self.vocab_size,
                                                              embedding_size=self.emb_dim,
                                                              output_projection=output_projection,
                                                              feed_previous=forward_only,
                                                              beam_search=beam_search,
                                                              beam_size=self.beam_size, #dtype=dtype
                                                              )

            def model_with_buckets(beam_search):

                if beam_search:
                    return rl_seq2seq.decode_model_with_buckets(encoder_inputs=self.encoder_inputs,
                                                                decoder_inputs=self.decoder_inputs, targets=targets,
                                                                weights=self.target_weights, buckets=self.buckets,
                                                                seq2seq=lambda x,y:seq2seq_f(x,y,True, True),
                                                                softmax_loss_function=softmax_loss_function)
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    if not beam_search:
                        return rl_seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets,
                                                             self.target_weights, self.buckets,
                                                             lambda x, y: seq2seq_f(x, y,
                                                                                tf.select(self.forward_only, True, False),
                                                                                tf.select(self.beam_search, True, False)),
                                                             # lambda x, y: seq2seq_f(x,y, False),
                                                             softmax_loss_function=softmax_loss_function)

            self.outputs, self.losses, self.encoder_states = tf.cond(self.beam_search, model_with_buckets(True),
                                                                     model_with_buckets(False))

            # if not beam_search:
            #     self.outputs, self.losses, self.encoder_states = rl_seq2seq.model_with_buckets(self.encoder_inputs,
            #                                         self.decoder_inputs, targets, self.target_weights, self.buckets,
            #                                     #lambda x, y: seq2seq_f(x, y, tf.select(self.forward_only, True, False)),
            #                                         lambda x, y: seq2seq_f(x,y, False),
            #                                          softmax_loss_function=softmax_loss_function)
            # else:
            #     self.outputs, self.probs, self.encoder_states = rl_seq2seq.decode_model_with_buckets(
            #         encoder_inputs=self.encoder_inputs, decoder_inputs=self.decoder_inputs, targets=targets,
            #         weights=self.target_weights, buckets=self.buckets, seq2seq=lambda x,y:seq2seq_f(x,y,True),
            #         softmax_loss_function=softmax_loss_function
            #     )

        # if forward_only and output_projection is not None:
        #     for b in xrange(len(self.buckets)):
        #         self.outputs[b] = [
        #             tf.matmul(output, output_projection[0]) + output_projection[1]
        #             for output in self.outputs[b]
        #         ]

        if not forward:
            with tf.name_scope("GRL_Gradient"):
                self.t_vars = [v for v in tf.trainable_variables() if name_scope in v.name]
                self.gradient_norms = []
                self.updatas = []

                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                for b in xrange(len(self.buckets)):
                    adjusted_losses = tf.mul(self.losses[b], self.rewards[b])
                    gradients = tf.gradients(adjusted_losses, self.t_vars)
                    clips_gradient, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    gradient_ops = opt.apply_gradients(zip(clips_gradient, self.t_vars), global_step=self.global_step)
                    self.updatas.append(gradient_ops)

        all_variables = [k for k in tf.global_variables() if name_scope in k.name]
        self.saver = tf.train.Saver(all_variables)

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, reward, bucket_id, forward_only, beam_search):
        encoder_size, decoder_size = self.buckets[bucket_id]

        input_feed = {self.forward_only.name: forward_only}
        input_feed = [self.beam_search.name] = beam_search
        for i in xrange(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in xrange(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]
        for i in xrange(len(self.buckets)):
            input_feed[self.rewards[i].name] = reward

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updatas[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
            updata, norm, loss = session.run(output_feed, input_feed)
            return updata, norm, loss
        else:
            # output_feed = [self.probs[bucket_id], self.encoder_states[bucket_id]]
            # for i in xrange(decoder_size):
            #     output_feed.append(self.outputs[bucket_id][i])

            output_feed = [self.probs[bucket_id], self.encoder_states[bucket_id], self.outputs[bucket_id]]

            outputs = session.run(output_feed, input_feed)  # loss, states, logits
            return outputs[0], outputs[1], outputs[2:]

    def step_rl(self, session, st_model, bk_model, encoder_inputs, decoder_inputs, target_weights,
                batch_source_encoder, bucket_id):
        init_inputs = [encoder_inputs, decoder_inputs, target_weights, bucket_id]

        batch_mask = [1 for _ in xrange(self.batch_size)]

        ep_rewards, ep_step_loss, enc_states = [], [], []
        ep_encoder_inputs, ep_target_weights, ep_bucket_id = [], [], []
        episode, dialog = 0, []

        while True:
            ep_encoder_inputs.append(batch_source_encoder)
            step_loss, encoder_states, output_logits = self.step(session, encoder_inputs, decoder_inputs, target_weights,
                                                                 reward=1, bucket_id=bucket_id, forward_only=True)
            ep_target_weights.append(target_weights)
            ep_bucket_id.append(bucket_id)
            ep_step_loss.append(step_loss)

            state_tran = np.transpose(encoder_states, axes=(1, 0, 2))
            print("state_tran: ", np.shape(state_tran))
            state_vec = np.reshape(state_tran, (self.batch_size, -1))
            print("state_vec: ", np.shape(state_vec))
            enc_states.append(state_vec)

            resp_tokens = self.remove_type(output_logits, self.buckets[bucket_id], type=1)

            try:
                encoder_trans = np.transpose(ep_encoder_inputs, axes=(1, 0))
            except ValueError:
                encoder_trans = np.transpose(ep_encoder_inputs, axes=(1, 0, 2))
            print ("[encoder_trans] shape: ", np.shape(encoder_trans))

            for i, (resp, ep_encoder) in enumerate(zip(resp_tokens, encoder_trans)):
                if (len(resp) <= 3) or (resp in self.dummy_dialogs) or (resp in ep_encoder.tolist()):
                    batch_mask[i] = 0
                    print("make mask index: %d, batch_mask: %s" % (i, batch_mask))
            if sum(batch_mask) == 0 or episode > 9:
                break

            # ----[Reward]----------------------------------------
            # r1: Ease of answering
            r1 = [self.logProb(session, st_model, self.buckets, resp_tokens, [d for _ in resp_tokens],
                               mask=batch_mask) for d in self.dummy_dialogs]
            print("r1: final value: ", r1)
            r1 = -np.mean(r1) if r1 else 0

            # r2: Information Flow
            r2_list = []
            if len(enc_states) < 4:
                r2 = 0
            else:
                batch_vec_a, batch_vec_b = enc_states[-3], enc_states[-1]
                for i, (vec_a, vec_b) in enumerate(zip(batch_vec_a, batch_vec_b)):
                    if batch_mask[i] == 0: continue
                    rr2 = sum(vec_a * vec_b) / sum(abs(vec_a) * abs(vec_b))
                    # print("vec_a*vec_b: %s" %sum(vec_a*vec_b))
                    # print("r2: %s" %r2)
                    if (rr2 < 0):
                        print("rr2: ", rr2)
                        print("vec_a: ", vec_a)
                        print("vec_b: ", vec_b)
                        rr2 = -rr2
                    else:
                        rr2 = -log(rr2)
                    r2_list.append(rr2)
                r2 = sum(r2_list) / len(r2_list)

            # r3: Semantic Coherence
            print("r3: Semantic Coherence")
            if len(ep_encoder_inputs) < 4:
                r3 = 0
            else:
                pi = ep_encoder_inputs[-3]
                qi = ep_encoder_inputs[-2]
                answer = ep_encoder_inputs[-1]
                query = np.column_stack((pi, qi))
                r3_1 = self.logProb(session, st_model, self.buckets, query, answer, mask=batch_mask)
                r3_2 = self.logProb(session, bk_model, self.buckets, answer, qi, mask=batch_mask)
                print("r3_1: ", r3_1)
                print("r3_2: ", r3_2)
                r3 = r3_1 + r3_2

            # Episode total reward
            print("r1: %s, r2: %s, r3: %s" % (r1, r2, r3))
            R = 0.25 * r1 + 0.25 * r2 + 0.5 * r3
            ep_rewards.append(R)
            # ----------------------------------------------------
            episode += 1

            # prepare for next dialogue
            bk_id = []
            for i in range(len(resp_tokens)):
                bk_id.append(min([b for b in range(len(self.buckets)) if self.buckets[b][0] >= len(resp_tokens[i])]))
            bucket_id = max(bk_id)
            feed_data = {bucket_id: [(resp_tokens, [])]}
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, _ = self.get_batch(feed_data,
                                                                                                     bucket_id, type=2)

        if len(ep_rewards) == 0:
            print("ep_rewards is zero")
            ep_rewards.append(1)

        print("[Step] final:", episode, ep_rewards)
        # gradient decent according to batch rewards
        # rto = 0.0
        # if (len(ep_step_loss) <= 1) or (len(ep_rewards) <= 1) or (max(ep_rewards) - min(ep_rewards) == 0):
        #     rto = 0.0
        # else:
        #     rto = (max(ep_step_loss) - min(ep_step_loss)) / (max(ep_rewards) - min(ep_rewards))
        # advantage = [np.mean(ep_rewards) * rto] * len(self.buckets)

        reward = [np.mean(ep_rewards)] * len(self.buckets)
        print("advantage: %s" % reward)
        updata, norm, loss = self.step(session, init_inputs[0], init_inputs[1], init_inputs[2], bucket_id=init_inputs[3],
                                    reward=reward, forward_only=False)

        return updata, norm, loss


    # log(P(|a)b), the conditional likelyhood
    def logProb(self, session, model, buckets, tokens_a, tokens_b, mask=None):
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        # prepare for next dialogue
        # bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(tokens_a) and buckets[b][1] > len(tokens_b)])
        # print("tokens_a: %s" %tokens_a)
        print("tokens_b: %s" % tokens_b)

        bk_id = []
        for i in xrange(len(tokens_a)):
            bk_id.append(min([b for b in xrange(len(buckets))
                              if buckets[b][0] >= len(tokens_a[i]) and buckets[b][1] >= len(tokens_b[i])]))
        bucket_id = max(bk_id)

        print("bucket_id: %s" % bucket_id)

        feed_data = {bucket_id: zip(tokens_a, tokens_b)}

        # print("logProb feed_back: %s" %feed_data[bucket_id])
        encoder_inputs, decoder_inputs, target_weights, _, _ = self.get_batch(feed_data, bucket_id, type=1)
        # print("logProb: encoder: %s; decoder: %s" %(encoder_inputs, decoder_inputs))
        # step
        _, _, output_logits = model.step(session, encoder_inputs, decoder_inputs, target_weights,
                                         bucket_id, forward_only=True, force_dec_input=True)

        logits_t = np.transpose(output_logits, (1, 0, 2))
        print("logits_t shape: ", np.shape(logits_t))

        sum_p = []
        for i, (tokens, logits) in enumerate(zip(tokens_b, logits_t)):
            print("tokens: %s, index: %d" % (tokens, i))
            # print("logits: %s" %logits)

            # if np.sum(tokens) == 0: break
            if mask[i] == 0: continue
            p = 1
            for t, logit in zip(tokens, logits):
                # print("logProb: logit: %s" %logit)
                norm = softmax(logit)[t]
                # print ("t: %s, norm: %s" %(t, norm))
                p *= norm
            if p < 1e-100:
                # print ("p: ", p)
                p = 1e-100
            p = log(p) / len(tokens)
            print ("logProb: p: %s" % (p))
            sum_p.append(p)
        re = np.sum(sum_p) / len(sum_p)
        # print("logProb: P: %s" %(re))
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

    def get_batch(self, train_data, bucket_id, type=0):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # print("Batch_Size: %s" %self.batch_size)
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_source_encoder, batch_source_decoder = [], []
        # print("bucket_id: %s" %bucket_id)
        for batch_i in xrange(self.batch_size):
            if type == 1:
                # feed_data = {bucket_id: zip(tokens_a, tokens_b)}
                encoder_input, decoder_input = train_data[bucket_id][batch_i]
            elif type == 2:
                # feed_data = {bucket_id: [(resp_tokens, [])]}
                encoder_input_a, decoder_input = train_data[bucket_id][0]
                encoder_input = encoder_input_a[batch_i]
            elif type == 0:
                encoder_input, decoder_input = random.choice(train_data[bucket_id])
                print("train en: %s, de: %s" % (encoder_input, decoder_input))

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

