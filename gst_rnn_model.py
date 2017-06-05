import tensorflow as tf
import gst_seq2seq as st_seq2seq
import numpy as np
import random
import data_utils

class gst_model(object):
    def __init__(self, gst_config, name_scope, forward_only = False, num_samples = 512, dtype=tf.float32):
        self.buckets = gst_config.buckets_concat
        self.emb_dim = gst_config.emb_dim
        self.batch_size = gst_config.batch_size
        self.vocab_size = gst_config.vocab_size
        #self.learning_rate = gst_config.learning_rate
        self.learning_rate = tf.Variable(initial_value=float(gst_config.learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * gst_config.learning_rate_decay_factor)

        max_gradient_norm = gst_config.max_gradient_norm
        num_layers = gst_config.num_layers

        with tf.name_scope("cell"):
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

        # the top of decoder_inputs is mark
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        softmax_loss_function = None
        output_projection = None
        if num_samples < self.vocab_size:
            w_t = tf.get_variable("proj_w", [self.vocab_size, self.emb_dim], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.vocab_size], dtype=dtype)
            output_projection = (w,b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                               num_samples, self.vocab_size),dtype)

            softmax_loss_function = sampled_loss

        with tf.name_scope("st_seq2seq"):
            def seq2seq_f(encoder_inputs, decoder_inputs, forward):
                return st_seq2seq.embedding_attention_seq2seq(encoder_inputs=encoder_inputs,
                                                              decoder_inputs=decoder_inputs,
                                                              cell=cells,
                                                              num_encoder_symbols=self.vocab_size,
                                                              num_decoder_symbols=self.vocab_size,
                                                              embedding_size=self.emb_dim,
                                                              output_projection=output_projection,
                                                              feed_previous=forward,
                                                              dtype=dtype)
            self.outputs, self.losses, _ = st_seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                                                         targets, self.target_weights, self.buckets,
                                                                         lambda x, y: seq2seq_f(x, y,
                                                                             tf.select(self.forward_only,True, False)),
                                                                         softmax_loss_function=softmax_loss_function)

            for b in xrange(len(self.buckets)):
                self.outputs[b] = [
                            tf.cond(
                                self.forward_only,
                                lambda: tf.matmul(output, output_projection[0]) + output_projection[1],
                                lambda: output
                            )
                            for output in self.outputs[b]
                        ]

        if not forward_only:
            with tf.name_scope("gst_radient"):
                self.t_vars = [v for v in tf.trainable_variables() if name_scope in v.name]
                self.gradient_norms = []
                self.updatas = []

                opt = tf.train.AdamOptimizer(learning_rate=0.001)
                #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                for b in xrange(len(self.buckets)):
                    gradients = tf.gradients(self.losses[b], self.t_vars)
                    clips_gradient, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    gradient_ops = opt.apply_gradients(zip(clips_gradient, self.t_vars), global_step=self.global_step)
                    self.updatas.append(gradient_ops)

        all_variables = [k for k in tf.global_variables() if name_scope in k.name]
        self.saver = tf.train.Saver(all_variables)

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]

        input_feed = {self.forward_only.name: forward_only}
        for i in xrange(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in xrange(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updatas[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
            updata, norm, loss = session.run(output_feed, input_feed)
            return updata, norm, loss
        else:
            output_feed = [self.outputs[bucket_id], self.losses[bucket_id]]
            output, loss = session.run(output_feed, input_feed)
            return output, loss

    def get_batch(self, train_data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        batch_source_encoder, batch_source_decoder = [], []

        #print("bucket_id: ", bucket_id)
        for batch_i in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(train_data[bucket_id])

            batch_source_encoder.append(encoder_input)
            batch_source_decoder.append(decoder_input)

            #print("encoder_input: ", encoder_input)
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            #print("encoder_input pad: ", list(reversed(encoder_input + encoder_pad)))

            #print("decoder_input: ", decoder_input)
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)
            #print("decoder_pad: ",[data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * decoder_pad_size)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

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