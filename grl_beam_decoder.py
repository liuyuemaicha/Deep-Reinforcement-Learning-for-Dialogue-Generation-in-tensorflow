import tensorflow as tf
from tensorflow.python.util import nest


def nest_map(func, nested):
    if not nest.is_sequence(nested):
        return func(nested)
    flat = nest.flatten(nested)
    return nest.pack_sequence_as(nested, list(map(func, flat)))


def sparse_boolean_mask(tensor, mask):
    """
    Creates a sparse tensor from masked elements of `tensor`

    Inputs:
      tensor: a 2-D tensor, [batch_size, T]
      mask: a 2-D mask, [batch_size, T]

    Output: a 2-D sparse tensor
    """
    mask_lens = tf.reduce_sum(tf.cast(mask, tf.int32), -1, keep_dims=True)
    mask_shape = tf.shape(mask)
    left_shifted_mask = tf.tile(
        tf.expand_dims(tf.range(mask_shape[1]), 0),
        [mask_shape[0], 1]
    ) < mask_lens
    return tf.SparseTensor(
        indices=tf.where(left_shifted_mask),
        values=tf.boolean_mask(tensor, mask),
        shape=tf.cast(tf.pack([mask_shape[0], tf.reduce_max(mask_lens)]), tf.int64)  # For 2D only
    )


def flat_batch_gather(flat_params, indices, validate_indices=None,
                      batch_size=None,
                      options_size=None):
    """
    Gather slices from `flat_params` according to `indices`, separately for each
    example in a batch.

    output[(b * indices_size + i), :, ..., :] = flat_params[(b * options_size + indices[b, i]), :, ..., :]

    The arguments `batch_size` and `options_size`, if provided, are used instead
    of looking up the shape from the inputs. This may help avoid redundant
    computation (TODO: figure out if tensorflow's optimizer can do this automatically)

    Args:
      flat_params: A `Tensor`, [batch_size * options_size, ...]
      indices: A `Tensor`, [batch_size, indices_size]
      validate_indices: An optional `bool`. Defaults to `True`
      batch_size: (optional) an integer or scalar tensor representing the batch size
      options_size: (optional) an integer or scalar Tensor representing the number of options to choose from
    """
    if batch_size is None:
        batch_size = indices.get_shape()[0].value
        if batch_size is None:
            batch_size = tf.shape(indices)[0]

    if options_size is None:
        options_size = flat_params.get_shape()[0].value
        if options_size is None:
            options_size = tf.shape(flat_params)[0] // batch_size
        else:
            options_size = options_size // batch_size

    indices_offsets = tf.reshape(tf.range(batch_size) * options_size, [-1] + [1] * (len(indices.get_shape()) - 1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)
    flat_indices_into_flat = tf.reshape(indices_into_flat, [-1])

    return tf.gather(flat_params, flat_indices_into_flat, validate_indices=validate_indices)


def batch_gather(params, indices, validate_indices=None,
                 batch_size=None,
                 options_size=None):
    """
    Gather slices from `params` according to `indices`, separately for each
    example in a batch.

    output[b, i, ..., j, :, ..., :] = params[b, indices[b, i, ..., j], :, ..., :]

    The arguments `batch_size` and `options_size`, if provided, are used instead
    of looking up the shape from the inputs. This may help avoid redundant
    computation (TODO: figure out if tensorflow's optimizer can do this automatically)

    Args:
      params: A `Tensor`, [batch_size, options_size, ...]
      indices: A `Tensor`, [batch_size, ...]
      validate_indices: An optional `bool`. Defaults to `True`
      batch_size: (optional) an integer or scalar tensor representing the batch size
      options_size: (optional) an integer or scalar Tensor representing the number of options to choose from
    """
    if batch_size is None:
        batch_size = params.get_shape()[0].merge_with(indices.get_shape()[0]).value
        if batch_size is None:
            batch_size = tf.shape(indices)[0]

    if options_size is None:
        options_size = params.get_shape()[1].value
        if options_size is None:
            options_size = tf.shape(params)[1]

    batch_size_times_options_size = batch_size * options_size

    # has no gradients implemented.
    flat_params = tf.reshape(params, tf.concat(0, [[batch_size_times_options_size], tf.shape(params)[2:]]))

    indices_offsets = tf.reshape(tf.range(batch_size) * options_size, [-1] + [1] * (len(indices.get_shape()) - 1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)

    return tf.gather(flat_params, indices_into_flat,validate_indices=validate_indices)


class BeamFlattenWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, beam_size):
        self.cell = cell
        self.beam_size = beam_size

    @staticmethod
    def merge_batch_beam(tensor):
        remaining_shape = tf.shape(tensor)[2:]
        res = tf.reshape(tensor, tf.concat(0, [[-1], remaining_shape]))
        res.set_shape(tf.TensorShape((None,)).concatenate(tensor.get_shape()[2:]))
        return res

    def unmerge_batch_beam(self, tensor):
        remaining_shape = tf.shape(tensor)[1:]
        res = tf.reshape(tensor, tf.concat(0, [[-1, self.beam_size], remaining_shape]))
        res.set_shape(tf.TensorShape((None, self.beam_size)).concatenate(tensor.get_shape()[1:]))
        return res

    def prepend_beam_size(self, element):
        return tf.TensorShape(self.beam_size).concatenate(element)

    def tile_along_beam(self, state):
        if nest.is_sequence(state):
            return nest_map(
                lambda val: self.tile_along_beam(val),
                state
            )

        if not isinstance(state, tf.Tensor):
            raise ValueError("State should be a sequence or tensor")

        tensor = state

        tensor_shape = tensor.get_shape().with_rank_at_least(1)
        new_tensor_shape = tensor_shape[:1].concatenate(self.beam_size).concatenate(tensor_shape[1:])

        dynamic_tensor_shape = tf.unstack(tf.shape(tensor))
        res = tf.expand_dims(tensor, 1)
        res = tf.tile(res, [1, self.beam_size] + [1] * (tensor_shape.ndims - 1))
        res = tf.reshape(res, [-1, self.beam_size] + list(dynamic_tensor_shape[1:]))
        res.set_shape(new_tensor_shape)
        return res

    def __call__(self, inputs, state, scope=None):
        flat_inputs = nest_map(self.merge_batch_beam, inputs)
        flat_state = nest_map(self.merge_batch_beam, state)

        flat_output, flat_next_state = self.cell(flat_inputs, flat_state, scope=scope)

        output = nest_map(self.unmerge_batch_beam, flat_output)
        next_state = nest_map(self.unmerge_batch_beam, flat_next_state)

        return output, next_state

    @property
    def state_size(self):
        return nest_map(self.prepend_beam_size, self.cell.state_size)

    @property
    def output_size(self):
        return nest_map(self.prepend_beam_size, self.cell.output_size)


class BeamReplicateWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, beam_size):
        self.cell = cell
        self.beam_size = beam_size

    def prepend_beam_size(self, element):
        return tf.TensorShape(self.beam_size).concatenate(element)

    def tile_along_beam(self, state):
        if nest.is_sequence(state):
            return nest_map(
                lambda val: self.tile_along_beam(val),
                state
            )

        if not isinstance(state, tf.Tensor):
            raise ValueError("State should be a sequence or tensor")

        tensor = state

        tensor_shape = tensor.get_shape().with_rank_at_least(1)
        new_tensor_shape = tensor_shape[:1].concatenate(self.beam_size).concatenate(tensor_shape[1:])

        dynamic_tensor_shape = tf.unstack(tf.shape(tensor))
        res = tf.expand_dims(tensor, 1)
        res = tf.tile(res, [1, self.beam_size] + [1] * (tensor_shape.ndims - 1))
        res = tf.reshape(res, [-1, self.beam_size] + list(dynamic_tensor_shape[1:]))
        res.set_shape(new_tensor_shape)
        return res

    def __call__(self, inputs, state, scope=None):
        varscope = scope or tf.get_variable_scope()

        flat_inputs = nest.flatten(inputs)
        flat_state = nest.flatten(state)

        flat_inputs_unstacked = list(zip(*[tf.unstack(tensor, num=self.beam_size, axis=1) for tensor in flat_inputs]))
        flat_state_unstacked = list(zip(*[tf.unstack(tensor, num=self.beam_size, axis=1) for tensor in flat_state]))

        flat_output_unstacked = []
        flat_next_state_unstacked = []
        output_sample = None
        next_state_sample = None

        for i, (inputs_k, state_k) in enumerate(zip(flat_inputs_unstacked, flat_state_unstacked)):
            inputs_k = nest.pack_sequence_as(inputs, inputs_k)
            state_k = nest.pack_sequence_as(state, state_k)

            if i == 0:
                output_k, next_state_k = self.cell(inputs_k, state_k, scope=scope)
            else:
                with tf.variable_scope(varscope, reuse=True):
                    output_k, next_state_k = self.cell(inputs_k, state_k, scope=varscope if scope is not None else None)

            flat_output_unstacked.append(nest.flatten(output_k))
            flat_next_state_unstacked.append(nest.flatten(next_state_k))

            output_sample = output_k
            next_state_sample = next_state_k

        flat_output = [tf.stack(tensors, axis=1) for tensors in zip(*flat_output_unstacked)]
        flat_next_state = [tf.stack(tensors, axis=1) for tensors in zip(*flat_next_state_unstacked)]

        output = nest.pack_sequence_as(output_sample, flat_output)
        next_state = nest.pack_sequence_as(next_state_sample, flat_next_state)

        return output, next_state

    @property
    def state_size(self):
        return nest_map(self.prepend_beam_size, self.cell.state_size)

    @property
    def output_size(self):
        return nest_map(self.prepend_beam_size, self.cell.output_size)


class BeamSearchHelper(object):
    INVALID_SCORE = -1e18  # top_k doesn't handle -inf well

    def __init__(self, cell, beam_size, stop_token, initial_state, initial_input,
                 max_len=200,
                 output_projection=None,
                 outputs_to_score_fn=None,
                 tokens_to_inputs_fn=None,
                 cell_transform='default',
                 scope=None):

        self.beam_size = beam_size
        self.stop_token = stop_token
        self.max_len = max_len
        self.scope = scope
        self.output_projection = output_projection

        if cell_transform == 'default':
            if type(cell) in [tf.nn.rnn_cell.LSTMCell,
                              tf.nn.rnn_cell.GRUCell,
                              tf.nn.rnn_cell.BasicLSTMCell,
                              tf.nn.rnn_cell.BasicRNNCell]:
                cell_transform = 'flatten'
            else:
                cell_transform = 'replicate'

        if cell_transform == 'none':
            self.cell = cell
            self.initial_state = initial_state
            self.initial_input = initial_input
        elif cell_transform == 'flatten':
            self.cell = BeamFlattenWrapper(cell, self.beam_size)
            self.initial_state = self.cell.tile_along_beam(initial_state)
            self.initial_input = self.cell.tile_along_beam(initial_input)
        elif cell_transform == 'replicate':
            self.cell = BeamReplicateWrapper(cell, self.beam_size)
            self.initial_state = self.cell.tile_along_beam(initial_state)
            self.initial_input = self.cell.tile_along_beam(initial_input)
        else:
            raise ValueError("cell_transform must be one of: 'default', 'flatten', 'replicate', 'none'")

        self._cell_transform_used = cell_transform

        if outputs_to_score_fn is not None:
            self.outputs_to_score_fn = outputs_to_score_fn
        if tokens_to_inputs_fn is not None:
            self.tokens_to_inputs_fn = tokens_to_inputs_fn

        batch_size = tf.Dimension(None)
        if not nest.is_sequence(self.initial_state):
            batch_size = batch_size.merge_with(self.initial_state.get_shape()[0])
        else:
            for tensor in nest.flatten(self.initial_state):
                batch_size = batch_size.merge_with(tensor.get_shape()[0])

        if not nest.is_sequence(self.initial_input):
            batch_size = batch_size.merge_with(self.initial_input.get_shape()[0])
        else:
            for tensor in nest.flatten(self.initial_input):
                batch_size = batch_size.merge_with(tensor.get_shape()[0])

        self.inferred_batch_size = batch_size.value
        if self.inferred_batch_size is not None:
            self.batch_size = self.inferred_batch_size
        else:
            if not nest.is_sequence(self.initial_state):
                self.batch_size = tf.shape(self.initial_state)[0]
            else:
                self.batch_size = tf.shape(list(nest.flatten(self.initial_state))[0])[0]

        self.inferred_batch_size_times_beam_size = None
        if self.inferred_batch_size is not None:
            self.inferred_batch_size_times_beam_size = self.inferred_batch_size * self.beam_size

        self.batch_size_times_beam_size = self.batch_size * self.beam_size

    @staticmethod
    def outputs_to_score_fn(cell_output):
        return tf.nn.log_softmax(cell_output)

    @staticmethod
    def tokens_to_inputs_fn(symbols):
        return tf.expand_dims(symbols, -1)

    def beam_setup(self, time):

        emit_output = None
        next_cell_state = self.initial_state
        next_input = self.initial_input

        # Set up the beam search tracking state
        cand_symbols = tf.fill([self.batch_size_times_beam_size, 0], tf.constant(self.stop_token, dtype=tf.int32))
        cand_logprobs = tf.ones((self.batch_size_times_beam_size,), dtype=tf.float32) * -float('inf')

        first_in_beam_mask = tf.equal(tf.range(self.batch_size_times_beam_size) % self.beam_size, 0)

        beam_symbols = tf.fill([self.batch_size_times_beam_size, 0], tf.constant(self.stop_token, dtype=tf.int32))
        beam_logprobs = tf.select(
            first_in_beam_mask,
            tf.fill([self.batch_size_times_beam_size], 0.0),
            tf.fill([self.batch_size_times_beam_size], self.INVALID_SCORE)
        )

        # Set up correct dimensions for maintaining loop invariants.
        # Note that the last dimension (initialized to zero) is not a loop invariant,
        # so we need to clear it.
        # inference so that _shape is not necessary?
        cand_symbols._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size, None))
        cand_logprobs._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size,))
        beam_symbols._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size, None))
        beam_logprobs._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size,))

        next_loop_state = (
            cand_symbols,
            cand_logprobs,
            beam_symbols,
            beam_logprobs,
        )

        emit_output = tf.zeros(self.cell.output_size)
        elements_finished = tf.zeros([self.batch_size], dtype=tf.bool)

        return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

    def beam_loop(self, time, cell_output, cell_state, loop_state):
        (
            past_cand_symbols,  # [batch_size*beam_size, time-1]
            past_cand_logprobs,  # [batch_size*beam_size]
            past_beam_symbols,  # [batch_size*beam_size, time-1], right-aligned
            past_beam_logprobs,  # [batch_size*beam_size]
        ) = loop_state

        # We don't actually use this, but emit_output is required to match the
        # cell output size specfication. Otherwise we would leave this as None.
        emit_output = cell_output

        cell_output = tf.unstack(cell_output, axis=1)

        if self.output_projection is not None:
            tmp = [tf.nn.xw_plus_b(output_k,
                                   self.output_projection[0],
                                   self.output_projection[1])
                   for output_k in cell_output]

        cell_output = tf.stack(tmp, axis=1)

        num_classes = int(cell_output.get_shape()[-1])

        # 1. Get scores for all candidate sequences
        logprobs = self.outputs_to_score_fn(cell_output)

        logprobs_batched = tf.reshape(logprobs + tf.expand_dims(tf.reshape(past_beam_logprobs, [self.batch_size, self.beam_size]), 2),
		                              [self.batch_size, self.beam_size * num_classes])

        # 2. Determine which states to pass to next iteration
        nondone_mask = tf.reshape(
            tf.cast(tf.equal(tf.range(num_classes), self.stop_token), tf.float32) * self.INVALID_SCORE,
            [1, 1, num_classes])

        nondone_mask = tf.reshape(tf.tile(nondone_mask, [1, self.beam_size, 1]),
                                  [-1, self.beam_size * num_classes])

        beam_logprobs, indices = tf.nn.top_k(logprobs_batched + nondone_mask, self.beam_size)
        beam_logprobs = tf.reshape(beam_logprobs, [-1])

        # For continuing to the next symbols
        symbols = indices % num_classes  # [batch_size, self.beam_size]
        parent_refs = indices // num_classes  # [batch_size, self.beam_size]

        symbols_history = flat_batch_gather(past_beam_symbols, parent_refs, batch_size=self.batch_size, options_size=self.beam_size)

        beam_symbols = tf.concat(1, [symbols_history, tf.reshape(symbols, [-1, 1])])

        # Handle the output and the cell state shuffling
        next_cell_state = nest_map(
            lambda element: batch_gather(element, parent_refs, batch_size=self.batch_size, options_size=self.beam_size),
            cell_state
        )

        next_input = self.tokens_to_inputs_fn(tf.reshape(symbols, [-1, self.beam_size]))

        # 3. Update the candidate pool to include entries that just ended with a stop token
        logprobs_done = tf.reshape(logprobs_batched, [-1, self.beam_size, num_classes])[:, :, self.stop_token]
        done_parent_refs = tf.argmax(logprobs_done, 1)
        done_symbols = flat_batch_gather(past_beam_symbols, done_parent_refs, batch_size=self.batch_size, options_size=self.beam_size)

        done_symbols = tf.tile(done_symbols, [self.beam_size, 1])

        logprobs_done_max = tf.reshape(logprobs_done, [-1])

        cand_symbols_unpadded = tf.select(logprobs_done_max > past_cand_logprobs,
                                          done_symbols,
                                          past_cand_symbols)
        cand_logprobs = tf.maximum(logprobs_done_max, past_cand_logprobs)

        cand_symbols = tf.concat(1, [cand_symbols_unpadded, tf.fill([self.batch_size_times_beam_size, 1], self.stop_token)])

        # 4. Check the stopping criteria
        # elements_finished = tf.reduce_max(tf.reshape(beam_logprobs, [-1, self.beam_size]), 1) < cand_logprobs
        finished = beam_logprobs < cand_logprobs
        elements_finished = tf.reduce_all(finished)
        elements_finished = tf.reshape(elements_finished, [self.batch_size])
        if self.max_len is not None:
            elements_finished_clip = (time >= self.max_len)
            elements_finished |= elements_finished_clip

        # 5. Prepare return values
        for tensor in list(nest.flatten(next_input)) + list(nest.flatten(next_cell_state)):
            tensor.set_shape(tf.TensorShape((self.inferred_batch_size, self.beam_size)).concatenate(tensor.get_shape()[2:]))

        for tensor in [cand_symbols, cand_logprobs, elements_finished]:
            tensor.set_shape(tf.TensorShape((self.inferred_batch_size,)).concatenate(tensor.get_shape()[1:]))

        for tensor in [beam_symbols, beam_logprobs]:
            tensor.set_shape(tf.TensorShape((self.inferred_batch_size_times_beam_size,)).concatenate(tensor.get_shape()[1:]))

        next_loop_state = (cand_symbols,
                           cand_logprobs,
                           beam_symbols,
                           beam_logprobs,)

        return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

    def loop_fn(self, time, cell_output, cell_state, loop_state):
        if cell_output is None:
            return self.beam_setup(time)
        else:
            return self.beam_loop(time, cell_output, cell_state, loop_state)

    def decode_dense(self):
        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(self.cell, self.loop_fn, scope=self.scope)

        cand_symbols, cand_logprobs, beam_symbols, beam_logprobs = final_loop_state
        return cand_symbols, cand_logprobs

    def decode_sparse(self, include_stop_tokens=True):
        dense_symbols, logprobs = self.decode_dense()
        mask = tf.not_equal(dense_symbols, self.stop_token)
        if include_stop_tokens:
            mask = tf.concat(1, [tf.ones_like(mask[:, :1]), mask[:, :-1]])
        return sparse_boolean_mask(dense_symbols, mask), logprobs


def beam_decoder(cell, beam_size, stop_token, initial_state, initial_input, tokens_to_inputs_fn,
                 output_projection=None,
                 outputs_to_score_fn=None,
                 max_len=None,
                 cell_transform='default',
                 output_dense=False,
                 scope=None):
    with tf.variable_scope(scope or "rnn_decoder") as varscope:
        helper = BeamSearchHelper(cell=cell,
                                  beam_size=beam_size,
                                  stop_token=stop_token,
                                  initial_state=initial_state,
                                  initial_input=initial_input,
                                  output_projection=output_projection,
                                  tokens_to_inputs_fn=tokens_to_inputs_fn,
                                  outputs_to_score_fn=outputs_to_score_fn,
                                  max_len=max_len,
                                  cell_transform=cell_transform,
                                  scope=varscope)

        if output_dense:
            return helper.decode_dense()
        else:
            return helper.decode_sparse()
