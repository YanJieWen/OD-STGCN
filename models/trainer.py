# @Time    : 2021/12/21 11:14
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : trainer
# @Project Name :new_code

import tensorflow as tf
import numpy as np
import time
from models.base_model import *
from data_loder.data_utils import *

def model_train(inputs_O,inputs_D,blocks,args,sum_path='./log'):
    n_zones,n_his,n_pred =args.n_zones,args.n_his,args.n_pred
    k_s,k_t = args.k_s,args.k_t#the size of kernel
    batch_size,epoch,opt =args.batch_size,args.epoch,args.opt

    #Declaration placeholder
    x_O = tf.placeholder(tf.float32,[None,n_his+1,n_zones,n_zones],name='input_original')
    x_D = tf.placeholder(tf.float32, [None, n_his+1, n_zones, n_zones], name='input_destination')
    # time_feature = tf.compat.v1.placeholder(tf.int32,[None,n_his+1,10])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    #define model
    train_loss,pred =intergrate_od_output(x_O, x_D, n_his, k_s, k_t, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)

    #learning rate setting
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs_O.shape[0]  # 训练数据集的长度9112
    if len_train % batch_size == 0:  # 遍历一次完整数据集的次数epoch_step
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)  # 将1赋值给global_steps
    # print([v.name for v in tf.all_variables()])

    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    # col = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    # for c in col:
    #     print('训练变量为:',c)

    with tf.Session() as sess:
        writer =tf.summary.FileWriter(sum_path, sess.graph)

        #run begin
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            start_time = time.time()
            for j,(O_batch,D_batch) in enumerate(gen_batch(inputs_O,inputs_D,batch_size,dynamic_batch=True, shuffle=True)):#by yield generator all data

                # print(np.shape(D_batch))
                _,summary = sess.run([train_op,merged], feed_dict={x_O: O_batch[:, 0:n_his + 1, :, :],x_D: D_batch[:, 0:n_his + 1, :, :],keep_prob: 1.0})
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x_O: O_batch[:, 0:n_his + 1, :, :],x_D: D_batch[:, 0:n_his + 1, :, :],
                                                                     keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')#per epoch to use time
            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, 'STGCN')
        writer.close()
        print('Training model finished!')


