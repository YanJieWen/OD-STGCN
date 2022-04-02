# @Time    : 2021/12/21 11:15
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : base_model
# @Project Name :new_code

import tensorflow as tf
from models.layers import *


def build_mode(inputs, n_his, k_s, k_t, blocks, keep_prob, data_attr):
    x = inputs[:, 0:n_his, :, :]#one for prediction
    ko = n_his  # to save the size of temproal conv
    for i, channels in enumerate(blocks):
        STGCN = Layers(x, k_s, k_t, channels, i, data_attr, keep_prob, act_func='GLU', tempro_model='NLSTM')
        x = STGCN.stgcn_module()
        ko -= 2 * (k_s - 1)
    if ko > 1:
        y = output_layer(x, ko, 'output_layer',data_attr)#when NLSTM or LSTM the k0 is constant=n_his(15)
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{ko}".')
    return y


def intergrate_od_output(o_inputs, d_inputs, n_his, k_s, k_t, blocks, keep_prob):
    y_original_pred = build_mode(o_inputs, n_his, k_s, k_t, blocks, keep_prob, data_attr='Orginal')
    y_destination_prd = build_mode(d_inputs, n_his, k_s, k_t, blocks, keep_prob, data_attr='Destinaltion')
    # o_pred_transpose = tf.transpose(y_original_pred, [0, 1, 3, 2])
    d_pred_tranpose = tf.transpose(y_destination_prd, [0, 1, 3, 2])
    # od_pred = (y_original_pred + y_destination_prd + o_pred_transpose + d_pred_tranpose) / 4#this will production symmetric matrix
    od_pred = tf.nn.sigmoid((y_original_pred+d_pred_tranpose)/2)
    # print(od_pred)
    # inputs = (o_inputs + tf.transpose(d_inputs, [0, 1, 3, 2])) / 2#this will production symmetric matrix
    inputs = o_inputs
    tf.add_to_collection(name='copy_loss',
                                   value=tf.nn.l2_loss(
                                       inputs[:, n_his-1:n_his, :, :] - inputs[:, n_his:n_his+1, :, :]))
    # print(inputs[:, n_his:n_his + 1, :, :])
    train_loss = tf.nn.l2_loss(od_pred - inputs[:, n_his:n_his + 1, :, :])
    single_pred =od_pred[:,0,:,:]#product the single time
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss,single_pred

def model_save(sess, global_steps, model_name):#save ckpt
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, save_path='./OD_GLU_model/model.ckpt', global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
