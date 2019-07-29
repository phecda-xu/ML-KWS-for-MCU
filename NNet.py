import tensorflow as tf
import math
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs


def create_dnn_model(fingerprint_input, model_settings, model_size_info,
                       is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    num_layers = len(model_size_info)
    layer_dim = [fingerprint_size]
    layer_dim.extend(model_size_info)
    flow = fingerprint_input
    tf.summary.histogram('input', flow)
    for i in range(1,num_layers+1):

        with tf.variable_scope('fc'+ str(i)):
            W = tf.get_variable('W', shape=[layer_dim[i-1], layer_dim[i]],
                                initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('fc_'+str(i)+'_w', W)
            b = tf.get_variable('b', shape=[layer_dim[i]])
            tf.summary.histogram('fc_'+str(i)+'_b', b)
            flow = tf.matmul(flow, W) + b

            flow = tf.nn.relu(flow)
            if is_training:
                flow = tf.nn.dropout(flow, dropout_prob)

    weights = tf.get_variable('final_fc', shape=[layer_dim[-1], label_count],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(tf.zeros([label_count]))

    logits = tf.matmul(flow,weights)+bias
    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_lstm_attentiion_model(fingerprint_input, model_settings, model_size_info,
                        is_training, time_major):
  """Builds a model with a lstm layer (with output projection layer and
       peep-hole connections)
  Based on model described in https://arxiv.org/abs/1705.02411
  model_size_info: [projection size, memory cells in LSTM]
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])

  num_classes = model_settings['label_count']
  projection_units = model_size_info[0]
  LSTM_units = model_size_info[1]
  attention_size = model_size_info[2]
  with tf.name_scope('LSTM-Layer'):
    with tf.variable_scope("lstm"):
      lstmcell = tf.contrib.rnn.LSTMCell(LSTM_units, use_peepholes=True,
                   num_proj=projection_units)
      output, last = tf.nn.dynamic_rnn(cell=lstmcell, inputs=fingerprint_4d,
                  dtype=tf.float32)
      flow = output

  if time_major:
      # (T,B,D) => (B,T,D)
      flow = tf.transpose(flow, [1, 0, 2])

  inputs_shape = flow.shape
  sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
  hidden_size = inputs_shape[2].value  # hidden size of the LSTM layer

  # Attention mechanism
  W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
  b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
  u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

  v = tf.tanh(tf.matmul(tf.reshape(flow, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
  vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
  exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
  alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

  # Output of LSTM is reduced with attention vector
  flow = tf.reduce_sum(flow * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

  with tf.name_scope('Output-Layer'):
    W_o = tf.get_variable('W_o', shape=[projection_units, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape=[num_classes])
    logits = tf.matmul(flow, W_o) + b_o

  if is_training:
    return logits, dropout_prob
  else:
    return logits


class LayerNormGRUCell(rnn_cell_impl.RNNCell):

  def __init__(self, num_units, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):

    super(LayerNormGRUCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      tf.logging.info("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args, copy):
    out_size = copy * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    with vs.variable_scope("gates"):
      h = state
      args = array_ops.concat([inputs, h], 1)
      concat = self._linear(args, 2)

      z, r = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
      if self._layer_norm:
        z = self._norm(z, "update")
        r = self._norm(r, "reset")

    with vs.variable_scope("candidate"):
      args = array_ops.concat([inputs, math_ops.sigmoid(r) * h], 1)
      new_c = self._linear(args, 1)
      if self._layer_norm:
        new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) * math_ops.sigmoid(z) + \
              (1 - math_ops.sigmoid(z)) * h
    return new_h, new_h


def create_crnn_attention_model(fingerprint_input, model_settings,
                      model_size_info, is_training, time_major=False):
    """Builds a model with convolutional recurrent networks with GRUs
    Based on the model definition in https://arxiv.org/abs/1703.05390
    model_size_info: defines the following convolution layer parameters
        {number of conv features, conv filter height, width, stride in y,x dir.},
        followed by number of GRU layers and number of GRU cells per layer
    Optionally, the bi-directional GRUs and/or GRU with layer-normalization
      can be explored.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    layer_norm = False
    bidirectional = False

    # CNN part
    first_filter_count = model_size_info[0]
    first_filter_height = model_size_info[1]
    first_filter_width = model_size_info[2]
    first_filter_stride_y = model_size_info[3]
    first_filter_stride_x = model_size_info[4]

    first_weights = tf.get_variable('W', shape=[first_filter_height,
                                                first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = int(math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x))
    first_conv_output_height = int(math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y))

    # GRU part
    num_rnn_layers = model_size_info[5]
    RNN_units = model_size_info[6]
    attention_size = model_size_info[6]
    flow = tf.reshape(first_dropout, [-1, first_conv_output_height,
                                      first_conv_output_width * first_filter_count])
    cell_fw = []
    cell_bw = []
    if layer_norm:
        for i in range(num_rnn_layers):
            cell_fw.append(LayerNormGRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(LayerNormGRUCell(RNN_units))
    else:
        for i in range(num_rnn_layers):
            cell_fw.append(tf.contrib.rnn.GRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(tf.contrib.rnn.GRUCell(RNN_units))

    if bidirectional:
        outputs, output_state_fw, output_state_bw = \
            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, flow,
                                                           dtype=tf.float32)
        flow_dim = first_conv_output_height * RNN_units * 2
        flow = tf.reshape(outputs, [-1, flow_dim])
    else:
        cells = tf.contrib.rnn.MultiRNNCell(cell_fw)
        outputs, last = tf.nn.dynamic_rnn(cell=cells, inputs=flow, dtype=tf.float32)
        flow_dim = RNN_units
        flow = outputs

    if time_major:
        # (T,B,D) => (B,T,D)
        flow = tf.transpose(flow, [1, 0, 2])

    inputs_shape = flow.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the LSTM layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(flow, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of LSTM is reduced with attention vector
    flow = tf.reduce_sum(flow * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    first_fc_output_channels = model_size_info[7]

    first_fc_weights = tf.get_variable('fcw', shape=[flow_dim,
                                                     first_fc_output_channels],
                                       initializer=tf.contrib.layers.xavier_initializer())

    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.nn.relu(tf.matmul(flow, first_fc_weights) + first_fc_bias)
    if is_training:
        final_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        final_fc_input = first_fc

    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, label_count], stddev=0.01))

    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


if __name__ == "__main__":
    model_settings = {'label_count': 3, "dct_coefficient_count": 13, "spectrogram_length": 320}
    model_size_info = [48, 10, 4, 2, 2, 2, 60, 84]
    fingerprint_input = tf.ones([model_settings["dct_coefficient_count"], model_settings["spectrogram_length"]])
    create_crnn_attention_model(fingerprint_input, model_settings, model_size_info, True)


