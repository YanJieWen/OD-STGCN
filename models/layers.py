# @Time    : 2021/12/21 11:15
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : layers
# @Project Name :new_code

import tensorflow as tf
from rnn_cell import NLSTMCell


class Layers():
    def __init__(self, x, k_s, k_t, channels, scope, data_attr, keep_prob, act_func='GLU', tempro_model='Conv_'):
        self.x = x
        self.k_s = k_s
        self.k_t = k_t
        self.channels = channels
        self.scope = scope
        self.keep_prob = keep_prob
        self.act_func = act_func
        self.tempro_model = tempro_model
        self.data_attr = data_attr

    def stgcn_module(self):  # the main layer
        c_si, c_t, c_oo = self.channels
        with tf.variable_scope(f'stn_block_{self.scope}_in{self.data_attr}'):
            x_s = tempro_layer(self.x, self.k_t, c_si, c_t, act_func=self.act_func,tempro_model=self.tempro_model)
            x_t = spatio_layer(x_s, self.k_s, c_t, c_t)
        with tf.variable_scope(f'stn_block_{self.scope}_out{self.data_attr}'):
            x_o = tempro_layer(x_t, self.k_t, c_t, c_oo,tempro_model=self.tempro_model)
        # print('the output is :',x_o)
        x_ln = layer_norm(x_o, f'layer_norm_{self.scope}_{self.data_attr}')
        return tf.nn.dropout(x_ln, self.keep_prob)


# the consist of STGCN Layers
def tempro_layer(x, k_t, c_in, c_out, act_func='relu', tempro_model='Conv_'):  # function is flexible
    _, T, n, _ = x.get_shape().as_list()
    if tempro_model == 'Conv_':
        if c_in > c_out:
            w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
            x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
        elif c_in < c_out:
            x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
        else:
            x_input = x
        x_input = x_input[:, k_t - 1:T, :, :]
        if act_func == 'GLU':  # 门控线性单元
            wt = tf.get_variable(name='wt', shape=[k_t, 1, c_in, 2 * c_out], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
            bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
            x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
            return x_conv[:, :, :, 0:c_out] + x_input * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])#GLU OR GTU%tanh
        else:
            wt = tf.get_variable(name='wt', shape=[k_t, 1, c_in, c_out], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
            bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
            x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
            if act_func == 'linear':
                return x_conv
            elif act_func == 'sigmoid':
                return tf.nn.sigmoid(x_conv)
            elif act_func == 'relu':
                return tf.nn.relu(x_conv + x_input)
            else:
                raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')
    elif tempro_model == 'LSTM':
        x_ = tf.transpose(x,[0,2,1,3])
        x_ = tf.reshape(x_, [-1, T, c_in])
        cell = tf.nn.rnn_cell.BasicLSTMCell(c_out)
        enc_outputs, enc_memory = tf.nn.dynamic_rnn(cell, x_, dtype=tf.float32)
        return tf.transpose(tf.reshape(enc_outputs, [-1,n, T, c_out]),[0,2,1,3])
    elif tempro_model =='NLSTM':
        x_ = tf.transpose(x,[0,2,1,3])
        x_ =tf.reshape(x_,[-1,T,c_in])
        cell = NLSTMCell(num_units=c_out,depth=2)
        enc_outputs, enc_memory= tf.nn.dynamic_rnn(cell, x_, dtype=tf.float32)
        return tf.transpose(tf.reshape(enc_outputs, [-1,n, T, c_out]),[0,2,1,3])
    else:
        raise ValueError(f'ERROR: model "{tempro_model}" is not defined.')


def spatio_layer(x, k_s, c_in, c_out):
    _, T, n, _ = x.get_shape().as_list()
    if c_in > c_out:
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x
    ws = tf.get_variable(name='ws', shape=[k_s * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    x_gconv = spectral_layer(tf.reshape(x, [-1, n, c_in]), ws, k_s, c_in, c_out) + bs
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


def spectral_layer(x, theta, k_s, c_in, c_out):
    kernel = tf.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, k_s, n])
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * k_s])
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


def layer_norm(x, scope):
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)
    # print('the mean {},and the sigma {}'.format(mu[0],sigma[0]))

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]),dtype=tf.float32)
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]),dtype=tf.float32)
        x_ = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return x_


def output_layer(x, T, scope, data_attr, act_func='GLU'):  # to get one temproal conv
    _, _, n, channel = x.get_shape().as_list()
    with tf.variable_scope(f'{scope}_in{data_attr}'):
        x_i = tempro_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}_{data_attr}')
    with tf.variable_scope(f'{scope}_out{data_attr}'):
        x_o = tempro_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    x_fc = fully_con_layer(x_o, n, channel, scope,data_attr)
    return x_fc


def fully_con_layer(x, n, channel, scope, data_attr):  # maps multi-channels to zones
    w = tf.get_variable(name=f'w_{scope}_{data_attr}', shape=[1, 1, channel, n], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}_{data_attr}', initializer=tf.zeros([n]), dtype=tf.float32)#
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    #use get_variable instead

def variable_summaries(var, v_name):  # summary to a tensor
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)
